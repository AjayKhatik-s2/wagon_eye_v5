# inputs/

Drop the 4 synchronized train videos here with these exact names:

    right_up.mp4         RIGHT_UP camera (master)
    left_up.mp4          LEFT_UP camera
    right_up_top.mp4     RIGHT_UP_TOP camera
    left_up_top.mp4      LEFT_UP_TOP camera

The 4 videos must already be trimmed to the same train pass so they
share a t=0 alignment.  If your filenames differ, either rename them or
pass --right_up / --left_up / --right_up_top / --left_up_top to
run_global_count.py.
