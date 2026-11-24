package main

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"
)

func testChatPrompt(t *testing.T, payload []byte, nativeID, workdir string) (string, error) {
	t.Helper()
	req, err := decodeChatRequest(payload)
	if err != nil {
		return "", err
	}
	return buildChatPrompt(req, nativeID, workdir)
}

func TestBuildPromptPreservesConversationRoles(t *testing.T) {
	payload := []byte(`{
      "model":"agy/default",
      "messages":[
        {"role":"system","content":"Be concise."},
        {"role":"user","content":"Hello"},
        {"role":"assistant","content":"Hi"},
        {"role":"user","content":[{"type":"text","text":"Explain Go."}]}
      ]
    }`)
	prompt, err := testChatPrompt(t, payload, "", "")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"role=system", "Be concise.", "role=user", "Explain Go."} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestBuildPromptResumedConversation(t *testing.T) {
	payload := []byte(`{
      "model":"agy/default",
      "conversation_id":"conv-abc",
      "messages":[
        {"role":"user","content":"First turn"},
        {"role":"assistant","content":"First answer"},
        {"role":"user","content":"Second turn"}
      ]
    }`)
	prompt, err := testChatPrompt(t, payload, "conv-abc", "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(prompt, "First turn") {
		t.Fatalf("resumed prompt should not duplicate prior user message:\n%s", prompt)
	}
	if strings.Contains(prompt, "First answer") {
		t.Fatalf("resumed prompt should not duplicate prior assistant message:\n%s", prompt)
	}
	if !strings.Contains(prompt, "Second turn") {
		t.Fatalf("resumed prompt missing latest user turn:\n%s", prompt)
	}
}

func TestBuildPromptStagesBase64Image(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cliproxy-img-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	fakePNG := base64.StdEncoding.EncodeToString([]byte("fake-png-content-data"))
	payload := []byte(`{
      "model":"agy/default",
      "messages":[
        {
          "role":"user",
          "content":[
            {"type":"text","text":"Look at this image"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,` + fakePNG + `"}}
          ]
        }
      ]
    }`)

	prompt, err := testChatPrompt(t, payload, "", tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "[Attached image:") || !strings.Contains(prompt, ".png") {
		t.Fatalf("expected staged image tag in prompt, got:\n%s", prompt)
	}
}

func TestBuildPromptInjectsClientToolsInstruction(t *testing.T) {
	payload := []byte(`{
      "model":"agy/default",
      "messages":[{"role":"user","content":"What is the weather?"}],
      "tools":[
        {
          "type":"function",
          "function":{"name":"get_weather","description":"Get current weather"}
        }
      ]
    }`)

	prompt, err := testChatPrompt(t, payload, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "[SYSTEM INSTRUCTION: CLIENT-SIDE TOOL CALLING]") {
		t.Fatalf("expected tool calling instructions in prompt:\n%s", prompt)
	}
	if !strings.Contains(prompt, "get_weather") {
		t.Fatalf("expected tool schema in prompt:\n%s", prompt)
	}
}

func TestExtractToolCalls(t *testing.T) {
	textWithFence := "Here is the tool call:\n```tool_calls\n[\n  {\n    \"name\": \"calc\",\n    \"arguments\": {\"expr\": \"2+2\"}\n  }\n]\n```"
	calls, ok := extractToolCalls(textWithFence)
	if !ok || len(calls) != 1 {
		t.Fatalf("failed to extract fenced tool calls: ok=%v, len=%d", ok, len(calls))
	}
	if calls[0].Function.Name != "calc" {
		t.Fatalf("call name = %q, want 'calc'", calls[0].Function.Name)
	}
	if !strings.Contains(calls[0].Function.Arguments, "2+2") {
		t.Fatalf("call args = %q", calls[0].Function.Arguments)
	}
}

func TestMakeChatCompletionRejectsEmptyResponse(t *testing.T) {
	for _, response := range []string{"", "   \n\t "} {
		_, err := makeChatCompletion("agy/gemini-4-argon-high", nil, agyResult{
			Status:   "SUCCESS",
			Response: response,
		})
		if err == nil {
			t.Fatalf("empty response %q produced a completion instead of an error", response)
		}
		var coded *pluginError
		if !errors.As(err, &coded) || coded.Code != "empty_assistant_response" {
			t.Fatalf("error = %v, want empty_assistant_response", err)
		}
		if coded.HTTPStatus != http.StatusBadGateway {
			t.Fatalf("status = %d, want 502", coded.HTTPStatus)
		}
	}
}

func TestMakeChatCompletionMapsToolCalls(t *testing.T) {
	raw, err := makeChatCompletion("gemini-3.8-flash-high", nil, agyResult{
		Status:   "SUCCESS",
		Response: "```tool_calls\n[{\"name\":\"search\",\"arguments\":{\"query\":\"golang\"}}]\n```",
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	choices := body["choices"].([]any)
	choice0 := choices[0].(map[string]any)
	if choice0["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v, want 'tool_calls'", choice0["finish_reason"])
	}
	msg := choice0["message"].(map[string]any)
	if msg["content"] != nil {
		t.Fatalf("content should be nil for tool_calls, got %v", msg["content"])
	}
	toolCalls := msg["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
}

func TestMakeChatCompletionMapsUsageAndConversation(t *testing.T) {
	raw, err := makeChatCompletion("agy/test", nil, agyResult{
		ConversationID: "conv-xyz-99",
		Status:         "SUCCESS",
		Response:       "done",
		Usage: agyUsage{
			InputTokens: 10, OutputTokens: 5, ThinkingTokens: 2, CacheReadTokens: 3, TotalTokens: 15,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "agy/test" {
		t.Fatalf("model = %#v", body["model"])
	}
	if body["conversation_id"] != "conv-xyz-99" {
		t.Fatalf("conversation_id = %#v, want 'conv-xyz-99'", body["conversation_id"])
	}
	choices := body["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	if message["content"] != "done" {
		t.Fatalf("content = %#v", message["content"])
	}
}

func TestMakeChatChunkIncludesConversationID(t *testing.T) {
	raw := makeChatChunk("id", 1, "agy/test", map[string]any{"content": "hello"}, nil, nil, "conv-stream-1")
	if !json.Valid(raw) || !strings.Contains(string(raw), `"conversation_id":"conv-stream-1"`) {
		t.Fatalf("expected conversation_id in JSON chunk: %s", raw)
	}
}

func TestMakeChatUsageChunkHasEmptyChoices(t *testing.T) {
	raw := makeChatChunk("id", 1, "agy/test", map[string]any{}, nil, map[string]any{"total_tokens": 1}, "")
	if !json.Valid(raw) || !strings.Contains(string(raw), `"choices":[]`) {
		t.Fatalf("unexpected usage chunk: %s", raw)
	}
}

func TestBuildPromptPutsToolInstructionBeforeMessages(t *testing.T) {
	payload := []byte(`{
      "model":"agy/default",
      "messages":[{"role":"user","content":"Hello"}],
      "tools":[{"type":"function","function":{"name":"bash","description":"Run commands"}}]
    }`)

	prompt, err := testChatPrompt(t, payload, "", "")
	if err != nil {
		t.Fatal(err)
	}
	toolIdx := strings.Index(prompt, "[SYSTEM INSTRUCTION: CLIENT-SIDE TOOL CALLING]")
	msgIdx := strings.Index(prompt, "--- message 1")
	if toolIdx < 0 || msgIdx < 0 {
		t.Fatalf("prompt is missing the tool instruction or the first message:\n%s", prompt[:200])
	}
	if toolIdx > msgIdx {
		t.Fatalf("tool instruction must precede the conversation so tail truncation cannot drop it")
	}
}

func TestWritePromptFileAndHandoff(t *testing.T) {
	dir := t.TempDir()
	big := strings.Repeat("conversation line\n", 20000) // ~340KB
	path, err := writePromptFile(dir, big)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Size() != int64(len(big)) {
		t.Fatalf("staged prompt size = %d, want %d", stat.Size(), len(big))
	}
	if !strings.Contains(path, ".cliproxy_prompts") {
		t.Fatalf("staged prompt path %q is not under .cliproxy_prompts", path)
	}

	wrapper := fileHandoffPrompt(path, len(big))
	if !strings.Contains(wrapper, path) {
		t.Fatalf("handoff prompt does not name the staged file: %s", wrapper)
	}
	if len(wrapper) > 4096 {
		t.Fatalf("handoff prompt is %d bytes, expected a small inline prompt", len(wrapper))
	}
}

func TestPromptBudgetDefault(t *testing.T) {
	if got := promptBudgetBytes(pluginConfig{}); got != defaultMaxPromptBytes {
		t.Fatalf("promptBudgetBytes() = %d, want %d", got, defaultMaxPromptBytes)
	}
	if got := promptBudgetBytes(pluginConfig{MaxPromptBytes: 32768}); got != 32768 {
		t.Fatalf("promptBudgetBytes(override) = %d, want 32768", got)
	}
}

// Models embed markdown fences inside string arguments, which used to cut the
// fenced tool_calls payload at the first inner fence and degrade it to text.
func TestExtractToolCallsWithEmbeddedFences(t *testing.T) {
	response := "```tool_calls\n[\n  {\n    \"name\": \"Agent\",\n    \"arguments\": {\n" +
		"      \"subagent_type\": \"worker\",\n" +
		"      \"prompt\": \"Please do this:\\n```bash\\nmake test\\n```\\nAnd also ```json\\n{\\\"a\\\":1}\\n```\\nthanks\"\n" +
		"    }\n  }\n]\n```"
	calls, ok := extractToolCalls(response)
	if !ok || len(calls) != 1 {
		t.Fatalf("expected one recovered tool call, got ok=%v len=%d", ok, len(calls))
	}
	if calls[0].Function.Name != "Agent" {
		t.Fatalf("name = %q, want Agent", calls[0].Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments are not valid JSON: %v (%s)", err, calls[0].Function.Arguments)
	}
	if args["subagent_type"] != "worker" {
		t.Fatalf("subagent_type = %v, want worker", args["subagent_type"])
	}
}

func TestExtractToolCallsPlainStillWorks(t *testing.T) {
	calls, ok := extractToolCalls("```tool_calls\n[{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}]\n```")
	if !ok || len(calls) != 1 || calls[0].Function.Name != "bash" {
		t.Fatalf("plain fenced call regressed: ok=%v calls=%v", ok, calls)
	}
}

func TestResolveToolNameSnapsModelAliases(t *testing.T) {
	declared := []string{"read", "bash", "subagent", "codemode"}
	cases := map[string]string{
		"subagent": "subagent",
		"AGENT":    "subagent", // model uses the human name for the tool
		"Bash":     "bash",
		"Codemode": "codemode",
		"write":    "write", // unknown stays untouched
	}
	for input, want := range cases {
		if got := resolveToolName(input, declared); got != want {
			t.Fatalf("resolveToolName(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestNormalizeToolCallsRewritesUnknownAlias(t *testing.T) {
	raw, err := makeChatCompletion("agy/test", []string{"read", "subagent"}, agyResult{
		Status:   "SUCCESS",
		Response: "```tool_calls\n[{\"name\":\"Agent\",\"arguments\":{\"agent\":\"scout\",\"task\":\"look\"}}]\n```",
	})
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Choices []struct {
			Message struct {
				ToolCalls []openAIToolCall `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Choices) == 0 || len(payload.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("no tool call in payload: %s", raw)
	}
	if got := payload.Choices[0].Message.ToolCalls[0].Function.Name; got != "subagent" {
		t.Fatalf("tool name = %q, want subagent", got)
	}
}

func TestToolInstructionListsExactNames(t *testing.T) {
	payload := []byte(`{"model":"agy/default","messages":[{"role":"user","content":"hi"}],
      "tools":[{"type":"function","function":{"name":"subagent","description":"delegate"}}]}`)
	prompt, err := testChatPrompt(t, payload, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "never a paraphrase") || !strings.Contains(prompt, "subagent") {
		t.Fatalf("tool instruction does not pin exact names:\n%s", prompt[:400])
	}
}
