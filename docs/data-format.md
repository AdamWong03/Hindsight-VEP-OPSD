# Training data format

Training data is a JSON array. Each item uses 1-based indices into the same
uniform pool of eight frames used by the compact VEP/TwiFF data:

```json
{
  "id": "example-id",
  "video": "videos/example.mp4",
  "image": [1, 2, 3, 4],
  "reasoning_image": [5, 6, 7, 8],
  "conversations": [
    {
      "from": "human",
      "value": "<image> <image> <image> <image> What happens next?"
    }
  ],
  "answer": "reference answer"
}
```

- `video` may be absolute. Relative paths are resolved below `VIDEO_ROOT`.
- `image` must contain observed-frame indices.
- `reasoning_image` must contain strictly later future-frame indices.
- The Student rollout, gradient recomputation, and formal inference use only
  `image`. The privileged fixed Teacher additionally receives `reasoning_image`.
- `answer` must be non-empty. It anchors the privileged Teacher semantically;
  it is not directly teacher-forced into the Student.
