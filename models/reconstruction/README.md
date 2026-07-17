# models/reconstruction/

Stage-1 (global wagon counting) YOLO weights.  Drop the 4 .pt files
here.  Short names are preferred; the legacy long names also work.

| Filename (preferred)        | Filename (legacy fallback)       | Used by               |
|-----------------------------|----------------------------------|-----------------------|
| `right_up_gap.pt`           | `right_up_wagon_gap.pt`          | RIGHT_UP (master)     |
| `left_up_gap.pt`            | `left_up_wagon_gap.pt`           | LEFT_UP               |
| `top_gap.pt`                | `top_gap.pt`                     | RIGHT_UP_TOP + LEFT_UP_TOP |
| `side_classification.pt`    | `side_classification.pt`         | RIGHT_UP (ENGINE / WAGON / BRAKE_VAN) |

The wagon_count subpackage looks up the preferred name first; if not
found, it tries the legacy name.  Either convention works.
