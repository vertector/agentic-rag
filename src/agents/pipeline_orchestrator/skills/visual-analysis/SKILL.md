---
name: visual-analysis
description: >
  Directs the orchestrator on how to handle visual chunks (tables, charts, figures)
  returned by the reranker. This skill is activated when a RETRIEVE result includes
  chunk_image_base64 and the user asks an analytical or descriptive question about it.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Visual Analysis Skill

## When to activate

Activate this skill when BOTH of the following are true:
1. One or more results in `reranker:agent_output` have a `chunk_type` of `table`, `chart`, `figure`, or `image`, AND contain the `chunk_image_base64` field.
2. The user's query requires an analytical interpretation of the visual content (e.g., "what does the chart show?", "extract the table values", "compare the columns", "what is the trend?", "find the value for X").

Do NOT activate this skill if the user is merely asking "find me a table about X" without asking for its contents to be analyzed. In that case, standard text citation is sufficient.

## How to use the image

When activated, you must:
1. Treat the `chunk_image_base64` field as an inline image provided in your context.
2. Use your native vision capabilities to "look" at the image.
3. Describe the visual content, extract data points from it, and answer the user's query directly based on what you see in the image.
4. Prioritize the insights derived from the image over the OCR-extracted text (`content`) or `summary`, as the visual layout often contains more accurate structured information than the text fallback.

## Formatting your response

When providing your answer:
- Lead with the insight or data extracted directly from the image.
- Cite the source chunk naturally.
- Explain the visual layout if it's relevant to the insight (e.g., "The bar chart shows..." or "In the fourth column of the table...").
- CRITICAL: ALWAYS extract and render the underlying raw chunk `content` (which holds the OCR-extracted markdown table or text) directly below your insight so the user can see the raw data! Do not suppress table outputs.

## Fallback behavior

If a visual chunk is returned but `chunk_image_base64` is missing or null:
- Do NOT mention that the image is missing or that you "cannot see" it.
- Seamlessly fall back to using the text in the `content` and `summary` fields to construct the best possible answer.

## Constraints

- NEVER expose or print the raw `chunk_image_base64` string to the user.
- NEVER describe the base64 string itself. You must interpret it as an image.
