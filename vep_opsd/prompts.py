from __future__ import annotations

import re
from typing import Sequence

from PIL import Image


STUDENT_SYSTEM_PROMPT = (
    "You are a video event prediction assistant. Predict what happens next using only "
    "the observed video frames and the question. Reason briefly, then commit to one "
    "specific future event. Never claim to have seen future frames and never refuse to predict."
)
TEACHER_SYSTEM_PROMPT = (
    "You are the privileged training-time teacher for video event prediction. You can see "
    "the observed frames, the ground-truth future frames, and the dataset's reference answer. "
    "Use the future as visual evidence and the reference answer as the semantic anchor. "
    "Independently reason toward the event that actually occurs. If the draft prefix is wrong, "
    "steer the continuation back toward the grounded event. Keep the required output format."
)
OUTPUT_FORMAT = (
    "Write brief reasoning inside <reason>...</reason>, followed by exactly one concise future "
    "event inside <answer>...</answer>."
)
TEACHER_INSTRUCTION = (
    "Question: {question}\n\n"
    "Dataset reference answer: <reference>{reference_answer}</reference>\n\n"
    "Continue as the assistant with an independent, future-grounded prediction. The continuation "
    "will be evaluated token by token on the current student's own draft prefix.\n\n"
    "{output_format}"
)


def clean_twiff_question(raw: str) -> str:
    text = re.sub(r"frame_\d+\s*", "", str(raw), flags=re.IGNORECASE)
    return " ".join(text.replace("<image>", " ").split()).strip()


def extract_final_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def student_messages(observed_frames: Sequence[Image.Image], question: str):
    content = [{"type": "image", "image": image} for image in observed_frames]
    content.append({"type": "text", "text": f"Question: {question}\n\n{OUTPUT_FORMAT}"})
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def privileged_teacher_messages(
    observed_frames: Sequence[Image.Image],
    future_frames: Sequence[Image.Image],
    question: str,
    reference_answer: str,
):
    content = [{"type": "text", "text": "Observed frames:"}]
    content.extend({"type": "image", "image": image} for image in observed_frames)
    content.append({"type": "text", "text": "Ground-truth future frames:"})
    content.extend({"type": "image", "image": image} for image in future_frames)
    content.append(
        {
            "type": "text",
            "text": TEACHER_INSTRUCTION.format(
                question=question,
                reference_answer=reference_answer,
                output_format=OUTPUT_FORMAT,
            ),
        }
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
