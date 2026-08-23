from wmfs.protocol import control as runtime
from wmfs_plugin import control as plugin


def test_runtime_and_plugin_control_golden_vectors_match() -> None:
    runtime_startup = runtime.Startup(
        0xFFFFFFFFFFFFFFFF,
        bytes(range(32)),
        bytes(reversed(range(32))),
        0x8877665544332211,
        255,
        6,
        11,
        1,
        b'{"unicode":"\xe2\x98\x83"}',
        (runtime.DescriptorRole.COMMAND_RING, runtime.DescriptorRole.FD_CONTROL),
        runtime.LogMode.CENTRALIZED,
    )
    plugin_startup = plugin.Startup(
        runtime_startup.session_generation,
        runtime_startup.interface_fingerprint,
        runtime_startup.configuration_fingerprint,
        runtime_startup.metadata_fingerprint,
        runtime_startup.capabilities,
        runtime_startup.operation_count,
        runtime_startup.protocol_version,
        runtime_startup.configuration_schema_version,
        runtime_startup.config,
        tuple(
            plugin.DescriptorRole(int(role))
            for role in runtime_startup.descriptor_roles
        ),
        plugin.LogMode.CENTRALIZED,
    )
    runtime_packet = runtime.encode_startup(runtime_startup, request_id=17)
    plugin_packet = plugin.encode_startup(plugin_startup, request_id=17)
    assert plugin_packet == runtime_packet
    assert plugin.decode_startup(runtime_packet)[1].config == runtime_startup.config
    assert runtime.decode_startup(plugin_packet)[1] == runtime_startup

    runtime_entries = (
        runtime.FdEntry(
            runtime.FdEntryKind.MAP,
            1,
            1 << 63,
            3,
            4,
            4096,
            runtime.FdFlag.WRITABLE | runtime.FdFlag.ARENA,
        ),
        runtime.FdEntry(runtime.FdEntryKind.RETIRE, 1, 1 << 63, 3, 4, 0),
    )
    plugin_entries = tuple(
        plugin.FdEntry(
            plugin.FdEntryKind(int(item.kind)),
            item.buffer_id,
            item.generation,
            item.allocation_id,
            item.invocation_id,
            item.byte_length,
            plugin.FdFlag(int(item.flags)),
        )
        for item in runtime_entries
    )
    assert runtime.encode_fd_batch(runtime.FdBatch(5, 6, runtime_entries)) == (
        plugin.encode_fd_batch(plugin.FdBatch(5, 6, plugin_entries))
    )
