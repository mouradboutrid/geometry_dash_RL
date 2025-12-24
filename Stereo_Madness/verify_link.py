import mmap
import ctypes
import time
import keyboard

# STRUCT DEFINITIONS (Match C++)
class ObjectData(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("type", ctypes.c_int),
    ]

class SharedState(ctypes.Structure):
    _fields_ = [
        ("cpp_writing", ctypes.c_int),
        ("py_writing", ctypes.c_int),
        ("player_x", ctypes.c_float),
        ("player_y", ctypes.c_float),
        ("player_vel_x", ctypes.c_float),
        ("player_vel_y", ctypes.c_float),
        ("player_rot", ctypes.c_float),
        ("gravity", ctypes.c_int),
        ("is_on_ground", ctypes.c_int),
        ("is_dead", ctypes.c_int),
        ("is_terminal", ctypes.c_int),
        ("percent", ctypes.c_float),
        ("dist_nearest_hazard", ctypes.c_float),
        ("dist_nearest_solid", ctypes.c_float),
        ("player_mode", ctypes.c_int),
        ("player_speed", ctypes.c_float),
        ("objects", ObjectData * 30),
        ("action_command", ctypes.c_int),
        ("reset_command", ctypes.c_int),
        ("checkpoint_command", ctypes.c_int),
    ]

def main():
    mem_name = "GD_RL_Memory"
    size = ctypes.sizeof(SharedState)
    
    try:
        shmem = mmap.mmap(-1, size, tagname=mem_name, access=mmap.ACCESS_DEFAULT)
        state = SharedState.from_buffer(shmem)
        print("Connected to GD Shared Memory.")
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # TEST STATE VARIABLES
    has_jumped_once = False
    checkpoints_hit = set()
    target_percentages = [10.0, 20.0, 30.0]

    print("Verification Loop Started...")
    
    try:
        while True:
            state.py_writing = 1 

            # READ STATS
            curr_pct = state.percent
            hazard_dist = state.dist_nearest_hazard
            is_dead = state.is_dead

            # LOGIC: JUMP ONLY ONCE FOR FIRST OBSTACLE
            if 0 < hazard_dist < 100 and not has_jumped_once:
                print(f"Obstacle detected at {hazard_dist:.1f}. Performing the single jump!")
                state.action_command = 1
                has_jumped_once = True
            else:
                # Release/stay idle after the first jump or if no hazard
                state.action_command = 0

            # LOGIC: CLICK 'W' AT PERCENTAGES
            for p in target_percentages:
                if curr_pct >= p and p not in checkpoints_hit:
                    print(f"Reached {p}%. Clicking 'W' for checkpoint.")
                    keyboard.press_and_release('w')
                    checkpoints_hit.add(p)

            # RESET ON DEATH
            if is_dead:
                if has_jumped_once:
                    print("Player died. Resetting jump flag for next attempt.")
                has_jumped_once = False
                checkpoints_hit.clear()

            # Unlock memory
            state.py_writing = 0 
            
            # Console Feedback
            print(f"Progress: {curr_pct:.1f}% | Hazard: {hazard_dist:.1f} | Jumped: {has_jumped_once}", end='\r')
            time.sleep(0.01) # ~100Hz

    except KeyboardInterrupt:
        print("\nTest terminated.")

main()
