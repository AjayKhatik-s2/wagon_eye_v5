# models/features/

Stage-3 feature inference YOLO weights.  Drop the 4 .pt files here.

| Filename                  | Used by                                   |
|---------------------------|-------------------------------------------|
| `door_state.pt`           | features/door — left + right door state   |
| `loaded.pt`               | features/load — LOADED / EMPTY            |
| `damage.pt`               | features/damage — top damage detection    |
| `wagon_id_counting.pt`    | features/ocr — wagon-number bbox detector |

The OCR processor additionally needs `easyocr` (installed via
`requirements.txt`).  No extra model files needed for easyocr — its
recognition models are auto-downloaded on first use.
