#!/bin/bash

unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY
unset KIMI_API_KEY
echo openai
echo $OPENAI_API_KEY
echo codex
echo $CODEX_API_KEY

codex --search exec --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"