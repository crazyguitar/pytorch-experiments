import io
import os
import sys

from contextlib import contextmanager
from typing import Generator, Union
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.filesystem import FileSystemBase

from s3path import S3Path
from s3torchconnector import S3Checkpoint


REGION = "us-west-2"


class S3FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs = None
        self.path = None

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        with S3Path(path).open(mode) as stream:
            yield stream

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        return path / suffix

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        self.fs = S3Checkpoint(region=REGION)
        self.path = S3Path(path)
        return path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        self.path.rename(path, new_path)

    def mkdir(self, path: [str, os.PathLike]) -> None:
        self.path.mkdir(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class S3WriteOnlyFileSystem(S3FileSystem):
    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        assert self.fs is not None
        print(f"s3:/{path}")
        with self.fs.writer(f"s3:/{path}") as stream:
            yield stream


class S3ReadOnlyFileSystem(S3FileSystem):
    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        assert self.fs is not None
        with self.fs.reader(f"s3:/{path}") as stream:
            yield stream


class S3Writer(FileSystemWriter):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
    ) -> None:
        super().__init__(
            path, single_file_per_rank, False, thread_count, per_thread_copy_ahead
        )
        self.fs = S3WriteOnlyFileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3WriteOnlyFileSystem.validate_checkpoint_id(checkpoint_id)


class S3Reader(FileSystemReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__(path)
        self.fs = S3ReadOnlyFileSystem()
        self.path = self.fs.init_path(path)


    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3ReadOnlyFileSystem.validate_checkpoint_id(checkpoint_id)


if __name__ == "__main__":
    # python fsdp_filesystem.py "/<bucket>/<key>"
    path = S3Path(sys.argv[1])
    w = S3Writer(path)
    w.reset()

    with w.fs.create_stream(path, "wb") as stream:
        stream.write(b"hello world")
