package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

type chatCompletionRequest struct {
	Model         string            `json:"model"`
	Messages      []chatMessage     `json:"messages"`
	Tools         []json.RawMessage `json:"tools,omitempty"`
	ToolChoice    any               `json:"tool_choice,omitempty"`
	Stream        bool              `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	conversationFields
	ReplaySummary string `json:"agy_replay_summary,omitempty"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Name       string          `json:"name,omitempty"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	Index    int                    `json:"index,omitempty"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openAIFunctionCallDesc `json:"function"`
}

type openAIFunctionCallDesc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCallCandidate struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func decodeChatRequest(raw []byte) (chatCompletionRequest, error) {
	var req chatCompletionRequest
	if len(raw) == 0 {
		return req, pError(errorCodeRequestScoped, "chat-completions payload is empty", http.StatusBadRequest)
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return req, pError(errorCodeRequestScoped, "invalid chat-completions JSON: "+err.Error(), http.StatusBadRequest)
	}
	if len(req.Messages) == 0 {
		return req, pError(errorCodeRequestScoped, "messages must contain at least one message", http.StatusBadRequest)
	}
	return req, nil
}

func addHandoffContext(req *chatCompletionRequest) error {
	if summary := strings.TrimSpace(req.ReplaySummary); summary != "" {
		content, _ := json.Marshal("Summary of the prior conversation:\n" + summary)
		req.Messages = append([]chatMessage{{Role: "user", Content: content}}, req.Messages...)
		return nil
	}
	for _, msg := range req.Messages[:len(req.Messages)-1] {
		if strings.EqualFold(strings.TrimSpace(msg.Role), "assistant") {
			return nil
		}
	}
	return pError(errorCodeRequestScoped, "Continuing on another account requires earlier chat messages or an agy_replay_summary, not just the latest turn.", http.StatusBadRequest)
}

func buildChatPrompt(req chatCompletionRequest, convID, workdir string) (string, error) {
	messages := req.Messages
	if convID != "" {
		// When resuming an existing conversation, only include messages after the last
		// assistant response to avoid duplicating history already stored in agy's session.
		lastAssistantIdx := -1
		for i := len(messages) - 1; i >= 0; i-- {
			if strings.EqualFold(strings.TrimSpace(messages[i].Role), "assistant") {
				lastAssistantIdx = i
				break
			}
		}
		if lastAssistantIdx >= 0 && lastAssistantIdx < len(messages)-1 {
			messages = messages[lastAssistantIdx+1:]
		}
	}

	// For a single user message in a resumed conversation without tools, pass the content directly.
	if convID != "" && len(messages) == 1 && (strings.TrimSpace(messages[0].Role) == "" || strings.EqualFold(strings.TrimSpace(messages[0].Role), "user")) && len(req.Tools) == 0 && messages[0].ToolCallID == "" && len(messages[0].ToolCalls) == 0 {
		content, err := messageContentText(messages[0].Content, workdir)
		if err != nil {
			return "", pError(errorCodeRequestScoped, fmt.Sprintf("message: %v", err), http.StatusBadRequest)
		}
		return content, nil
	}

	var b strings.Builder
	if convID == "" {
		b.WriteString("You are answering a request forwarded through an OpenAI-compatible chat-completions bridge to the official Antigravity CLI.\n")
		b.WriteString("Follow the system and developer instructions in the conversation below, then answer the latest user request. Return only the assistant response.\n")
		b.WriteString("You have no shell, editor, or network tools for this request: do not modify files, run commands, or inspect the environment. If the caller points you at a request file, read only that file. Answer from the conversation text alone, and never leave the reply empty.\n\n")
	}

	// The client-tools contract must never be lost, so it goes first: agy caps the
	// inline prompt size and truncates from the tail.
	if len(req.Tools) > 0 {
		writeToolInstruction(&b, req.Tools)
	}

	for i, msg := range messages {
		role := strings.TrimSpace(strings.ToLower(msg.Role))
		if role == "" {
			role = "user"
		}
		content, err := messageContentText(msg.Content, workdir)
		if err != nil {
			return "", pError(errorCodeRequestScoped, fmt.Sprintf("message %d: %v", i+1, err), http.StatusBadRequest)
		}

		b.WriteString("--- message ")
		b.WriteString(fmt.Sprintf("%d", i+1))
		b.WriteString(" role=")
		b.WriteString(role)
		if msg.Name != "" {
			b.WriteString(" name=")
			b.WriteString(msg.Name)
		}
		if msg.ToolCallID != "" {
			b.WriteString(" tool_call_id=")
			b.WriteString(msg.ToolCallID)
		}
		b.WriteString(" ---\n")
		if content != "" {
			b.WriteString(content)
			if !strings.HasSuffix(content, "\n") {
				b.WriteByte('\n')
			}
		}
		if len(msg.ToolCalls) > 0 && string(msg.ToolCalls) != "null" && string(msg.ToolCalls) != "[]" {
			b.WriteString("Prior assistant tool calls:\n")
			b.Write(msg.ToolCalls)
			b.WriteByte('\n')
		}
	}

	return b.String(), nil
}

// defaultMaxPromptBytes keeps inline prompts below the Antigravity CLI cap
// (observed around 190KB) so nothing has to be truncated before it is sent.
const defaultMaxPromptBytes = 176 * 1024

// writeToolInstruction renders the client-side tool calling contract.
func writeToolInstruction(b *strings.Builder, tools []json.RawMessage) {
	b.WriteString("[SYSTEM INSTRUCTION: CLIENT-SIDE TOOL CALLING]\n")
	b.WriteString("The client application has provided the following tools:\n")
	for _, tool := range tools {
		b.Write(tool)
		b.WriteByte('\n')
	}
	var names []string
	for _, tool := range tools {
		var parsed struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if err := json.Unmarshal(tool, &parsed); err == nil && strings.TrimSpace(parsed.Function.Name) != "" {
			names = append(names, parsed.Function.Name)
		}
	}
	if len(names) > 0 {
		b.WriteString("\nCall tools by these exact names (case-sensitive), never a paraphrase such as \"Agent\" for \"subagent\": ")
		b.WriteString(strings.Join(names, ", "))
		b.WriteString("\n")
	}
	b.WriteString("\nIf a client tool call is needed, reply ONLY with a JSON block in this exact markdown code fence:\n")
	b.WriteString("```tool_calls\n[\n  {\n    \"name\": \"tool_name\",\n    \"arguments\": { ... }\n  }\n]\n```\n")
	b.WriteString("If no tool call is needed, provide your regular assistant response without the ```tool_calls code block.\n")
	b.WriteString("Do not execute local commands for client tools; the client application will run them and return the result.\n\n")
}

// promptBudgetBytes is the inline stdin budget. Anything larger is handed to agy
// as a file so the full conversation reaches the model untruncated.
func promptBudgetBytes(cfg pluginConfig) int {
	if cfg.MaxPromptBytes > 0 {
		return cfg.MaxPromptBytes
	}
	return defaultMaxPromptBytes
}

// writePromptFile stores an oversized prompt where agy can read it with its own
// file tooling, which is not capped like the inline prompt.
func writePromptFile(workdir, prompt string) (string, error) {
	dir := filepath.Join(os.TempDir(), "cliproxy_prompts")
	if workdir != "" {
		dir = filepath.Join(workdir, ".cliproxy_prompts")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(prompt))
	path := filepath.Join(dir, "prompt_"+hex.EncodeToString(sum[:8])+".txt")
	if err := os.WriteFile(path, []byte(prompt), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// fileHandoffPrompt is the small inline prompt used with a written request file.
func fileHandoffPrompt(path string, promptBytes int) string {
	return fmt.Sprintf("READ THIS FIRST: this turn's complete client request (%d bytes) is stored in the file %s .\n"+
		"Read that file in full before answering: use your file-reading tool with offset/limit and keep reading until you have consumed every line.\n"+
		"The file begins with a [SYSTEM INSTRUCTION: CLIENT-SIDE TOOL CALLING] block that defines the client's tools, followed by the conversation transcript and the latest user request.\n"+
		"Follow those instructions exactly: answer the latest user request, and when a client tool call is required reply ONLY with the fenced tool_calls JSON block described there.\n"+
		"Do not modify files, run local commands, or inspect the machine beyond reading that file.\n", promptBytes, path)
}

func stageImage(dataURLOrPath string, workdir string) (string, error) {
	dataURLOrPath = strings.TrimSpace(dataURLOrPath)
	if dataURLOrPath == "" {
		return "", fmt.Errorf("empty image URL or path")
	}

	if strings.HasPrefix(dataURLOrPath, "file://") {
		return strings.TrimPrefix(dataURLOrPath, "file://"), nil
	}
	if filepath.IsAbs(dataURLOrPath) {
		if _, err := os.Stat(dataURLOrPath); err == nil {
			return dataURLOrPath, nil
		}
	}

	if strings.HasPrefix(dataURLOrPath, "data:image/") {
		commaIdx := strings.IndexByte(dataURLOrPath, ',')
		if commaIdx < 0 {
			return "", fmt.Errorf("invalid data URL: missing comma")
		}
		header := dataURLOrPath[:commaIdx]
		encodedData := dataURLOrPath[commaIdx+1:]

		subtype := "png"
		if strings.Contains(header, "image/jpeg") || strings.Contains(header, "image/jpg") {
			subtype = "jpg"
		} else if strings.Contains(header, "image/webp") {
			subtype = "webp"
		} else if strings.Contains(header, "image/gif") {
			subtype = "gif"
		}

		rawBytes, err := base64.StdEncoding.DecodeString(encodedData)
		if err != nil {
			return "", fmt.Errorf("failed to decode base64 image: %w", err)
		}

		cacheDir := filepath.Join(os.TempDir(), "cliproxy_images")
		if workdir != "" {
			cacheDir = filepath.Join(workdir, ".cliproxy_cache", "images")
		}
		if err := os.MkdirAll(cacheDir, 0700); err != nil {
			return "", fmt.Errorf("failed to create image cache directory: %w", err)
		}

		hash := sha256.Sum256(rawBytes)
		filename := fmt.Sprintf("img_%s.%s", hex.EncodeToString(hash[:8]), subtype)
		targetPath := filepath.Join(cacheDir, filename)

		if _, err := os.Stat(targetPath); os.IsNotExist(err) {
			if err := os.WriteFile(targetPath, rawBytes, 0600); err != nil {
				return "", fmt.Errorf("failed to write cached image: %w", err)
			}
		}
		return targetPath, nil
	}

	return dataURLOrPath, nil
}

func messageContentText(raw json.RawMessage, workdir string) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}

	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("content must be a string or an array of content blocks")
	}

	var parts []string
	for _, block := range blocks {
		kind, _ := block["type"].(string)
		switch kind {
		case "text", "input_text", "output_text", "":
			if text, ok := block["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
		case "image_url", "input_image", "image":
			imgURL := ""
			if imgMap, ok := block["image_url"].(map[string]any); ok {
				imgURL, _ = imgMap["url"].(string)
			} else if srcMap, ok := block["source"].(map[string]any); ok {
				if b64Data, ok := srcMap["data"].(string); ok {
					mediaType, _ := srcMap["media_type"].(string)
					if mediaType == "" {
						mediaType = "image/png"
					}
					imgURL = fmt.Sprintf("data:%s;base64,%s", mediaType, b64Data)
				}
			} else if strURL, ok := block["url"].(string); ok {
				imgURL = strURL
			}

			if imgURL != "" {
				stagedPath, err := stageImage(imgURL, workdir)
				if err != nil {
					return "", fmt.Errorf("staging image: %w", err)
				}
				parts = append(parts, fmt.Sprintf("[Attached image: %s (inspect this image file to answer the request)]", stagedPath))
			}
		default:
			if text, ok := block["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "\n"), nil
}

// jsonValueEnd returns the index just past the JSON object or array that starts
// at start, honouring strings and escapes. Tool arguments often embed markdown
// fences or braces, so the first "```" after the opening fence is not a safe
// terminator.
func jsonValueEnd(s string, start int) int {
	if start < 0 || start >= len(s) {
		return -1
	}
	var open, close byte
	switch s[start] {
	case '{':
		open, close = '{', '}'
	case '[':
		open, close = '[', ']'
	default:
		return -1
	}
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// fencedToolCallJSON pulls the JSON payload out of a fenced tool_calls block.
// Candidates are tried from most to least permissive so embedded ``` fences
// inside string arguments do not truncate the payload.
func fencedToolCallJSON(text string) string {
	start := -1
	for _, opener := range []string{"```tool_calls", "```json"} {
		if idx := strings.Index(text, opener); idx >= 0 {
			start = idx + len(opener)
			break
		}
	}
	if start < 0 {
		return ""
	}
	body := text[start:]
	if brace := strings.IndexAny(body, "[{"); brace >= 0 {
		if end := jsonValueEnd(body, brace); end > 0 {
			if candidate := strings.TrimSpace(body[brace:end]); candidate != "" {
				return candidate
			}
		}
	}
	// Fall back to the last closing fence, which is correct when the arguments
	// contain no balanced JSON the scanner could follow.
	if endIdx := strings.LastIndex(body, "```"); endIdx >= 0 {
		return strings.TrimSpace(body[:endIdx])
	}
	return ""
}

// declaredToolNames lists the tool names the client advertised.
func declaredToolNames(payload []byte) []string {
	var req struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		return nil
	}
	names := make([]string, 0, len(req.Tools))
	for _, raw := range req.Tools {
		var tool struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if err := json.Unmarshal(raw, &tool); err == nil && strings.TrimSpace(tool.Function.Name) != "" {
			names = append(names, tool.Function.Name)
		}
	}
	return names
}

// resolveToolName snaps a model-emitted name onto one of the advertised tools.
// Models routinely call "Agent" for an advertised "subagent", or change case,
// and an unknown name makes the client drop the whole call.
func resolveToolName(name string, declared []string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" || len(declared) == 0 {
		return trimmed
	}
	for _, candidate := range declared {
		if candidate == trimmed {
			return candidate
		}
	}
	lowered := strings.ToLower(trimmed)
	for _, candidate := range declared {
		if strings.ToLower(candidate) == lowered {
			return candidate
		}
	}
	matched := ""
	for _, candidate := range declared {
		lowerCandidate := strings.ToLower(candidate)
		if strings.HasSuffix(lowerCandidate, lowered) || strings.HasSuffix(lowered, lowerCandidate) {
			if matched != "" && matched != candidate {
				return trimmed // ambiguous, leave it alone
			}
			matched = candidate
		}
	}
	if matched != "" {
		return matched
	}
	return trimmed
}

func normalizeToolCalls(calls []openAIToolCall, declared []string) []openAIToolCall {
	if len(declared) == 0 {
		return calls
	}
	for i := range calls {
		calls[i].Function.Name = resolveToolName(calls[i].Function.Name, declared)
	}
	return calls
}

func extractToolCalls(response string) ([]openAIToolCall, bool) {
	trimmed := strings.TrimSpace(response)
	jsonBlock := fencedToolCallJSON(trimmed)

	if jsonBlock == "" {
		if strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
			jsonBlock = trimmed
		}
	}

	if jsonBlock == "" {
		return nil, false
	}

	var candidates []toolCallCandidate
	if err := json.Unmarshal([]byte(jsonBlock), &candidates); err != nil {
		var single toolCallCandidate
		if errSingle := json.Unmarshal([]byte(jsonBlock), &single); errSingle == nil && single.Name != "" {
			candidates = []toolCallCandidate{single}
		} else {
			return nil, false
		}
	}

	if len(candidates) == 0 {
		return nil, false
	}

	var calls []openAIToolCall
	for i, c := range candidates {
		if strings.TrimSpace(c.Name) == "" {
			continue
		}
		args := string(c.Arguments)
		if strings.TrimSpace(args) == "" || args == "null" {
			args = "{}"
		}
		calls = append(calls, openAIToolCall{
			Index: i,
			ID:    fmt.Sprintf("call_%s_%d", newCallID(), i),
			Type:  "function",
			Function: openAIFunctionCallDesc{
				Name:      c.Name,
				Arguments: args,
			},
		})
	}

	if len(calls) == 0 {
		return nil, false
	}
	return calls, true
}

// emptyResponseError marks a turn where agy finished successfully without any
// assistant text. The CLI usually spent that turn running its own built-in
// tools, so the bridge has nothing to forward and an empty completion would make
// clients retry the identical request in a loop. Retryable: a fresh attempt, or
// a fallback model, can still answer.
func emptyResponseError() error {
	return pRetryableError(
		"empty_assistant_response",
		"agy returned no assistant text: it likely executed its own tools instead of answering. Retry the request or switch models.",
		http.StatusBadGateway,
	)
}

func newCallID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err == nil {
		return hex.EncodeToString(buf)
	}
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func requestWantsUsage(raw []byte) bool {
	req, err := decodeChatRequest(raw)
	return err == nil && req.StreamOptions.IncludeUsage
}

func estimateTokens(text string) int {
	n := utf8.RuneCountInString(text)
	if n == 0 {
		return 0
	}
	return (n + 3) / 4
}

func makeChatCompletion(requestedModel string, declared []string, result agyResult) ([]byte, error) {
	model := displayModel(requestedModel)
	toolCalls, isToolCall := extractToolCalls(result.Response)
	toolCalls = normalizeToolCalls(toolCalls, declared)
	if !isToolCall && strings.TrimSpace(result.Response) == "" {
		return nil, emptyResponseError()
	}

	var message map[string]any
	var finishReason string
	if isToolCall {
		finishReason = "tool_calls"
		message = map[string]any{
			"role":       "assistant",
			"content":    nil,
			"tool_calls": toolCalls,
		}
	} else {
		finishReason = "stop"
		message = map[string]any{
			"role":    "assistant",
			"content": result.Response,
		}
	}

	payload := map[string]any{
		"id":      newCompletionID(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": openAIUsage(result.Usage),
	}
	if result.ConversationID != "" {
		payload["conversation_id"] = result.ConversationID
	}
	return json.Marshal(payload)
}

func openAIUsage(usage agyUsage) map[string]any {
	total := usage.TotalTokens
	if total == 0 {
		total = usage.InputTokens + usage.OutputTokens
	}
	out := map[string]any{
		"prompt_tokens":     usage.InputTokens,
		"completion_tokens": usage.OutputTokens,
		"total_tokens":      total,
	}
	if usage.ThinkingTokens > 0 {
		out["completion_tokens_details"] = map[string]any{"reasoning_tokens": usage.ThinkingTokens}
	}
	if usage.CacheReadTokens > 0 {
		out["prompt_tokens_details"] = map[string]any{"cached_tokens": usage.CacheReadTokens}
	}
	return out
}

func displayModel(model string) string {
	if strings.TrimSpace(model) == "" {
		return "agy/default"
	}
	return model
}

func newCompletionID() string {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err == nil {
		return "chatcmpl-agy-" + hex.EncodeToString(buf)
	}
	return fmt.Sprintf("chatcmpl-agy-%d", time.Now().UnixNano())
}

func makeChatChunk(id string, created int64, model string, delta map[string]any, finishReason *string, usage map[string]any, conversationID string) []byte {
	choice := map[string]any{
		"index": 0,
		"delta": delta,
	}
	if finishReason == nil {
		choice["finish_reason"] = nil
	} else {
		choice["finish_reason"] = *finishReason
	}
	body := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []any{choice},
	}
	if strings.TrimSpace(conversationID) != "" {
		body["conversation_id"] = strings.TrimSpace(conversationID)
	}
	if usage != nil {
		body["choices"] = []any{}
		body["usage"] = usage
	}
	raw, _ := json.Marshal(body)
	return raw
}
