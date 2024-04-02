from accelerate import Accelerator
from accelerate.state import AcceleratorState


def main():
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")


if __name__ == "__main__":
    main()