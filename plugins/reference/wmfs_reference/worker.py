from wmfs_reference.plugin import plugin


def main() -> None:
    from wmfs_plugin import worker_main

    worker_main(plugin)


if __name__ == "__main__":
    main()
