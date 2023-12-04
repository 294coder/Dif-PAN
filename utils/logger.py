import time
from functools import partial
import sys

import shortuuid
from torch import Tensor
import tensorboardX.writer as writer
from beartype import beartype
import os
import logging


def place_exists(place):
    return os.path.exists(place)


def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


class TensorboardLogger:
    def __init__(self, place="./runs/", file_dir='./logs/', file_logger_name=None, random_id=True, tb_comment=None,
                 **tb_kwargs) -> None:
        """tensorboard logger

        Args:
            place (str, optional): place to save tb logging. Defaults to './tb_runs/'.
            file_logger_name (str, optional): name of file logger. Defaults to None.
            random_id: (bool, optional): append id after tensorboard dir. Defaults to True
            tb_kwargs: kwargs for tensorboardX.writer.SummaryWriter
                comment: Optional[str] = "",
                purge_step: Optional[int] = None,
                max_queue: Optional[int] = 10,
                flush_secs: Optional[int] = 120,
                filename_suffix: Optional[str] = '',
                write_to_disk: Optional[bool] = True,
                log_dir: Optional[str] = None,
                comet_config: Optional[dict] = {"disabled": True}, 
        """
        if not place_exists(place):
            os.mkdir(place)
            print("Created directory: ", place)

        if random_id:
            # use random id as dir name
            id = generate_id(8) + '' if tb_comment is None else '_{}'.format(tb_comment)
            place = os.path.join(place, id)
            os.mkdir(place)
        else:
            # use time as dir name
            stf_time = time.strftime("%m-%d %H:%M", time.localtime())
            name = stf_time + '' if tb_comment is None else '_{}'.format(tb_comment)
            place = os.path.join(place, name)
            os.mkdir(place)

        self.writer = writer.SummaryWriter(place, **tb_kwargs)

        assert file_logger_name is not None, '@file_logger_name should be set'
        self.file_writer = PrintLogger(os.path.join(file_dir, file_logger_name + '.log'))

        self._tb_print = partial(self.file_writer.log, level=logging.DEBUG)

    def print(self, msg: str, level=logging.INFO):
        self.file_writer.log(msg, level)

    @beartype
    def log_scalar(self, tag: str, value: float, step: int):
        self._tb_print(f"add tb scalar {tag}: {value}")
        self.writer.add_scalar(tag, value, step)

    @beartype
    def log_scalars(self, tag: str, values: dict, step: int, on_one_fig: bool = False):
        self._tb_print(f"add tb scalars {tag}: {values}")
        if not on_one_fig:
            for t, v in values.items():
                self.writer.add_scalar(tag + '/' + t, v, step)
        else:
            self.writer.add_scalars(tag, values)

    @beartype
    def log_image(self, tag: str, image: Tensor, step: int):
        self._tb_print(f"add tb image {tag}: {image.shape}")
        self.writer.add_image(tag, image, step)

    @beartype
    def log_images(
            self, tag: str, images: Tensor, step: int, *, dataformats: str = "NCHW"
    ):
        self._tb_print(f"add tb images {tag}: {images.shape}")
        self.writer.add_images(tag, images, step, dataformats=dataformats)


class PrintLogger:
    def __init__(self, place, level=logging.DEBUG) -> None:
        assert place.endswith(".log"), "@place should be a file path"

        self.place = place

        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        
        _dir = os.path.dirname(place)
        assert os.path.exists(_dir), f"dir {_dir} not exists"
        
        file_handler = logging.FileHandler(place)
        file_handler.setLevel(level)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            r"%(asctime)s - %(levelname)s - %(message)s",
            datefmt=r"%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.init_print()

    def init_print(self):
        self.log('log will print out in {}'.format(self.place))

    def log(self, msg: str, level=logging.INFO):
        self.logger.log(level, msg)

    def print(self, *args, **kwargs):
        self.log(*args, *kwargs)


if __name__ == '__main__':
    logger = PrintLogger(place='./test_log.log')
    logger.log('this is a test log', level=logging.DEBUG)
