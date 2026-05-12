import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import cast

from datasets import load_dataset  # type: ignore[reportUnknownVariableType]

from saffron.eval.gsm8k import extract_answer
from saffron.model.vllm_teacher import VLLMTeacher, VLLMTeacherConfig

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([\d,.-]+)\}")


def _normalize_answer(text: str) -> str:
    # Rewrite \\boxed{X} → #### X so the output matches GSM8K format.
    # The teacher model did not respect the prompt formatting request.
    return _BOXED_RE.sub(lambda m: f"#### {m.group(1)}", text)


def main(
    teacher_config: VLLMTeacherConfig,
    dataset: str,
    name: str | None,
    split: str,
    question_field: str,
    answer_field: str,
    output_path: Path,
    batch_size: int,
    max_rounds: int,
) -> None:
    teacher = VLLMTeacher(teacher_config)
    ds = cast(list[dict[str, str]], load_dataset(dataset, name, split=split))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track a correct teacher answer per example index
    answered: dict[int, str] = {}

    # pending: list of (original_index, example)
    pending = list(enumerate(ds))

    start_time = time.time()
    for round_num in range(max_rounds):
        if not pending:
            break

        prev_answered = len(answered)

        # Generate in batches
        still_pending: list[tuple[int, dict[str, str]]] = []
        n_batches = (len(pending) + batch_size - 1) // batch_size
        for batch_num, batch_start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[batch_start : batch_start + batch_size]
            questions = [ex[question_field] for _, ex in batch]
            answers = [_normalize_answer(a) for a in teacher.generate(questions)]
            batch_correct = 0
            for (idx, ex), answer in zip(batch, answers, strict=True):
                gt = extract_answer(ex[answer_field])
                pred = extract_answer(answer)
                logger.info(
                    f"real: {gt} | tried: {pred} | len: {len(answer)} vs {len(ex[answer_field])}"
                )
                if gt is not None and pred == gt:
                    answered[idx] = answer
                    batch_correct += 1
                else:
                    still_pending.append((idx, ex))
            logger.info(
                f"Round {round_num + 1}/{max_rounds} | batch {batch_num}/{n_batches} | "
                f"+{batch_correct}/{len(batch)} correct | {len(answered)}/{len(ds)} total"
            )

        pending = still_pending
        round_delta = len(answered) - prev_answered
        logger.info(
            f"Round {round_num + 1}/{max_rounds} complete: "
            f"+{round_delta} this round, {len(answered)}/{len(ds)} total"
        )

    elapsed_min = (time.time() - start_time) / 60
    logger.info(
        f"Done in {elapsed_min:.1f} min: {len(answered)} answered by teacher, "
        f"{len(ds) - len(answered)} falling back to original"
    )

    total_written_len = 0
    total_original_len = 0
    with output_path.open("w") as f:
        for idx, ex in enumerate(ds):
            answer = answered.get(idx, ex[answer_field])
            total_written_len += len(answer)
            total_original_len += len(ex[answer_field])
            f.write(json.dumps({**ex, answer_field: answer}) + "\n")

    n = len(ds)
    avg_written = total_written_len / n if n else 0
    avg_original = total_original_len / n if n else 0
    logger.info(
        f"Avg answer len — written: {avg_written:.0f} chars | original: {avg_original:.0f} chars"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)[args.key]

    teacher_config = VLLMTeacherConfig(
        model_name=cfg["teacher"]["model_name"],
        max_tokens=cfg["teacher"]["max_tokens"],
        temperature=cfg["teacher"]["temperature"],
        dtype=cfg["teacher"]["dtype"],
        system_prompt=cfg["teacher"]["system_prompt"],
    )

    main(
        teacher_config=teacher_config,
        dataset=cfg["dataset"]["name"],
        name=cfg["dataset"]["subset"],
        split=cfg["dataset"]["split"],
        question_field=cfg["dataset"]["question_field"],
        answer_field=cfg["dataset"]["answer_field"],
        output_path=Path(cfg["output_path"]),
        batch_size=cfg["batch_size"],
        max_rounds=cfg["max_rounds"],
    )
