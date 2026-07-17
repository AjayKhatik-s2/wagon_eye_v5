# models/

Drop the 4 YOLO weights here with these exact names:

    right_up_wagon_gap.pt     gap detection on RIGHT_UP (master)

    left_up_wagon_gap.pt      gap detection on LEFT_UP

    top_gap.pt                gap detection on TOP cameras
                              (RIGHT_UP_TOP, LEFT_UP_TOP)

    side_classification.pt    ENGINE / WAGON / BRAKE_VAN classifier
                              (RIGHT_UP only — master authority)
