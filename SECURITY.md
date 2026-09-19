# Security

oat-notes runs entirely on-device. Its security promises are:

- Captured audio, transcripts, dictated text and voice embeddings never leave
  the machine and never reach a durable log.
- The only network traffic is the one-time model download, and `--offline`
  disables that.
- The web UI binds to `127.0.0.1` only and rejects requests whose
  `Host`/`Origin` headers name anything else.
- Dictation writes to the clipboard with Win+V history and cloud sync excluded.

A bug that breaks any of these is a security issue, even if nothing "attacks"
anything.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/Mason-Levyy/oat-notes/security/advisories/new)
rather than a public issue. Include the version (commit or installer name),
the Windows build and the steps to reproduce; do not include real transcripts
or audio.

You will get an acknowledgement within 7 days and a fix or a decision within
30. Please give that window before disclosing publicly.

## Supported versions

Only the latest release and `main` receive fixes.
