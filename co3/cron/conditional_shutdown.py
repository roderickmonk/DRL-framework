import logging
import os

import psutil


def process_running(substring: str)-> bool:
    """
    Check if there is any running process whose name contains the substring.
    """
    for proc in psutil.process_iter():

        try:
            command_line = proc.cmdline()

            # Check if process name contains the given substring.
            if len(command_line) > 1 and substring.lower() in command_line[1]:
                # print(f"{command_line=}")
                return True

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


if __name__ == "__main__":

    try:
        playback_active = process_running('co3.py')
        generate_active = process_running('generate.py')
        print(f"Playback Active: {playback_active}")
        print(f"Generate Active: {generate_active}")
        logging.error(f"Generate Active: {generate_active}")

        if not playback_active and not generate_active:

            auto_shutdown = bool(os.environ.get("AUTO_SHUTDOWN"))
            
            if auto_shutdown:
                print("Shutting down...")
                ret = os.system("sudo shutdown -h --no-wall")
            else:
                print("Practice shutdown...")
                ret = os.system("sudo shutdown -k --no-wall")

            print(f"Shutdown return status: {ret}")

        else:
            print("Shutdown not triggered")

    except KeyboardInterrupt:
        logging.fatal("KeyboardInterrupt")

    except Exception as err:
        logging.fatal(err)

    else:
        logging.warning("✨That's✨All✨Folks✨")
