# Security policy

Please report a vulnerability privately to Dongfang Zhao <dzhao@cs.washington.edu> instead of opening a public issue.
Include the affected version, a description, and a way to reproduce it. You will receive an acknowledgment within
five working days, and the fix will be released together with a note in `CHANGELOG.md`.

Only the latest release receives security fixes.

An index directory is trusted input: `load` reads the arrays that `meta.json` names, including absolute paths, so
load only directories that you or a trusted builder wrote.
