import signal
import sys
import  os

def handle_signal(signum, frame):
    signal_names = {
        signal.SIGINT: "SIGINT",
        signal.SIGHUP: "SIGHUP",
        signal.SIGTERM: "SIGTERM"
    }
    signal_name = signal_names.get(signum, "UNKNOWN")
    print(f"Received signal: {signal_name} ({signum}). Exiting gracefully.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

print("Signal trap script is running. Press Ctrl+C to send SIGINT, or send SIGHUP/SIGTERM to the process.")
print(f"PID: {sys.argv[0]} is {os.getpid()}")

# Keep the script running to catch signals
try:
    while True:
        pass
except KeyboardInterrupt:
    # This ensures the handler catches it, but prevents unhandled exceptions
    pass
