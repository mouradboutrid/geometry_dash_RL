import mmap
import ctypes
import time
from config import MEM_NAME

# C++ STRUCT MAPPING
# Must match the Struct definition in my Geode C++ Mod (utridu) exactly.
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
        # Synchronization Flags
        ("cpp_writing", ctypes.c_int),
        ("py_writing", ctypes.c_int),
        
        # Player Physics
        ("player_x", ctypes.c_float),
        ("player_y", ctypes.c_float),
        ("player_vel_x", ctypes.c_float),
        ("player_vel_y", ctypes.c_float),
        ("player_rot", ctypes.c_float),
        ("gravity", ctypes.c_int),
        ("is_on_ground", ctypes.c_int),
        ("is_dead", ctypes.c_int),
        ("is_terminal", ctypes.c_int),
        
        # Reward Data
        ("percent", ctypes.c_float),
        ("dist_nearest_hazard", ctypes.c_float),
        ("dist_nearest_solid", ctypes.c_float),
        ("player_mode", ctypes.c_int),  # 0 = Cube, 1 = Ship
        ("player_speed", ctypes.c_float),
        
        # Environment
        ("objects", ObjectData * 30),
        
        # Commands
        ("action_command", ctypes.c_int),     # 0=Release, 1=Hold
        ("reset_command", ctypes.c_int),      # 1=Reset Level
        ("checkpoint_command", ctypes.c_int), # 1=Set Checkpoint (Not used often, I use 'W'(Practice mod in the game ...))
    ]

class MemoryBridge:
    def __init__(self):
        try:
            # Connect to existing shared memory created by C++
            self.shmem = mmap.mmap(-1, ctypes.sizeof(SharedState), tagname=MEM_NAME, access=mmap.ACCESS_DEFAULT)
            self.state = SharedState.from_buffer(self.shmem)
            print(f"[MemoryBridge] Successfully connected to '{MEM_NAME}'")
        except FileNotFoundError:
            raise Exception(f"[Critical] Could not find Shared Memory '{MEM_NAME}'.\n"
                            "Make sure Geometry Dash is running with the Mod installed.")

    def read_state(self):
        """
        Reads the current state from shared memory.
        Uses a spinlock to avoid reading while C++ is writing.
        """
        # Safety break to prevent infinite freeze if C++ crashes while writing
        timeout = 0
        while self.state.cpp_writing == 1:
            timeout += 1
            if timeout > 2000: break # Break lock if stuck
            
        self.state.py_writing = 1  # Lock for Python
        
        # We return the raw state object. 
        # In a highly optimized version, we might copy specific fields here.
        return self.state

    def write_action(self, action: int):
        """
        Writes the action command to memory.
        """
        self.state.action_command = int(action)
        self.state.py_writing = 0  # Unlock

    def send_reset(self):
        """
        Signals the C++ mod to reset the level.
        """
        self.state.py_writing = 1
        self.state.reset_command = 1
        self.state.action_command = 0 # Ensure player doesn't jump immediately
        self.state.py_writing = 0
        time.sleep(0.05) # Give C++ time to process

    def close(self):
        self.shmem.close()
