API_KEY=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
curl https://api.openai.com/v1/engines/text-davinci-001/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer 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' \
  -d '{
  "temperature": 0,
  "prompt": "Q: Imagine there is a bet with a 60% chance of winning $100 and a 40% chance of losing $100. Should you take the bet?\nA:",
  "max_tokens": 100,
  "logprobs": 100
}'
