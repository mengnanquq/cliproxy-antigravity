# Fork notes

This fork (`mengnanquq/cliproxy-antigravity`) tracks
[`adeebahmad01/cliproxy-antigravity`](https://github.com/adeebahmad01/cliproxy-antigravity)
directly. There is no fork-only patch series any more: every fix this fork
carried has been merged upstream, and the one change upstream did not adopt was
measured and found unnecessary on current `agy` builds.

Tracking base: upstream `main` at `15e509d` (`v0.7.1` + the release-artifact
version fix from #45).

## History

| Fork work | Status |
| --- | --- |
| Client-tools contract placed before the conversation (#26) | upstream |
| Balanced-JSON tool-call parsing (#27) | upstream |
| Model catalog retained when `agy models` fails (#28) | upstream |
| NDJSON events up to 32MB, emit failures reported (#29, #31) | upstream |
| Tool names snapped onto advertised tools (#30) | upstream |
| Prompt staging for oversized requests, incl. private staging root and platform-specific privacy validation (#34) | upstream (`max_prompt_bytes`, `max_prompt_action`) |
| `count_tokens` measured on the built conversation | upstream |
| Stream keepalive | superseded by upstream `stream_keepalive_interval` (v0.7.0) |
| Keepalive started with the process instead of with `init` (`a6da725`) | **dropped**, see below |

## Why the keepalive-earlier change was dropped

`a6da725` started the keepalive ticker at process spawn because `agy` was
observed to stay silent past the proxies' ~30s idle window *before* emitting
`init`, notably on staged turns. Re-measured against `agy 1.5.17`
(`--output-format stream-json`, direct CLI):

| Case | time to `init` | first assistant text | whole turn |
| --- | --- | --- | --- |
| staged handoff, 300,089-byte request | 2.86s | 16.6s | 41.8s |
| inline prompt, 140KB | 2.77s | — | — |
| inline prompt, 300KB | 3.01s | — | — |

`agy` emits `init` before it reads the staged file, so the pre-`init` window is
binary startup, not prompt size. The long silences are all post-`init` (largest
gap observed: 17.5s) and are covered by `stream_keepalive_interval`, whose
ticker starts on `init` by design: the first empty delta commits the stream, so
an error arriving after one interval can no longer fall back through the host's
retry and account rotation.

The old behaviour (ticker at spawn) is preserved on the `archive/keepalive-at-spawn`
branch at `a6da725`, and the last pre-rebase fork build on
`archive/v0.5.1-fork.2`, if an older `agy` build still needs either. If a
version-sensitive fallback ever becomes necessary again, prefer a post-spawn
watchdog (keep the `init` gate, but start the ticker when no event arrived
within one interval of spawn) over starting it unconditionally — see
[mengnanquq/cliproxy-antigravity#1](https://github.com/mengnanquq/cliproxy-antigravity/issues/1).

## Building

Same as upstream; see `README.md`. Run the tests with a real `agy` off `PATH`:

```sh
go build ./... && go test ./...
```

`TestMain` refuses to run when a real `agy` is reachable, so hide it first (a
container, or `unshare -Ur -m sh -c 'mount --bind /dev/null /usr/bin/agy && go test ./...'`).
