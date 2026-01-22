import threading
import subprocess, time


def run_step(proc, command, timeout):
    """
    Send a command to a running bash process and capture its output,
    with a per-command timeout.
    """
    result = {'out': '', 'err': '', 'done': False}

    def reader():
        proc.stdin.write(command + "\n")
        proc.stdin.flush()

        # Read until a unique marker appears
        marker = "__END_OF_CMD__"
        proc.stdin.write(f"echo {marker}\n")
        proc.stdin.flush()

        out_lines = []
        for line in proc.stdout:
            if marker in line:
                break
            out_lines.append(line)
        result['out'] = ''.join(out_lines)
        result['done'] = True

    thread = threading.Thread(target=reader)
    thread.start()
    thread.join(timeout)

    if not result['done']:
        proc.kill()
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout, output=result['out'], stderr=result['err'])

    return result['out'], result['err']
