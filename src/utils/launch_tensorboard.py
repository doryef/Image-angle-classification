import sys
import socket
from tensorboard import program
import webbrowser

def get_ip_address():
    """Get the machine's IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def launch_tensorboard(logdir='logs', port=6006, host=None):
    """Launch TensorBoard server with remote access enabled"""
    ip_address = host if host else get_ip_address()
    
    # Remove any existing TensorBoard instances
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--bind_all'])
    if host:
        tb.configure(argv=[None, '--logdir', logdir, '--host', host])
    url = tb.launch()
    
    print(f"\nTensorBoard is running locally at: {url}")
    print(f"Access remotely at: http://{ip_address}:{port}")
    print("\nTo access TensorBoard from another computer:")
    print(f"1. Make sure you can reach {ip_address} from the remote computer")
    print("2. Open a web browser on the remote computer")
    print(f"3. Navigate to: http://{ip_address}:{port}")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        tb.main()
    except KeyboardInterrupt:
        print("\nShutting down TensorBoard server...")

if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6006
    host = sys.argv[3] if len(sys.argv) > 3 else None
    launch_tensorboard(logdir, port)