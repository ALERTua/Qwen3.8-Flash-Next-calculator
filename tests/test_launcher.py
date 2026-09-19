"""The launcher .cmd script: building it and fetching its assets.

Run with:  uv run pytest tests/test_launcher.py
"""

from qwen_flash_calculator import estimator as E
from qwen_flash_calculator import measured as M


def _grid_config(ctx: int, ubatch: int, fit_margin: int | None = None) -> E.Config:
    """The conditions the 24-run table was measured under.

    ``fit_margin`` defaults to the 512 MiB the table was measured with. The
    interface opens at 1024, so the grid is also checked there.

    Also defined in test_estimator.py: two test classes there use it too,
    and a shared module would be one more file to keep in step for a
    handful of lines.
    """
    return E.Config(
        ctx=ctx,
        ubatch=ubatch,
        kv_type="q8_0",
        threads=M.THREAD_REFERENCE,
        fit_margin=(M.REFERENCE_FIT_MARGIN_MIB if fit_margin is None else fit_margin),
        vram_total_mib=M.REFERENCE_VRAM_MIB,
    )


class TestLauncherScript:
    """The .cmd text built for the panel at the bottom of the window."""

    def test_every_set_line_carries_the_configured_value(self):
        cfg = E.Config(
            ctx=163840,
            ubatch=512,
            kv_type="f16",
            threads=13,
            fit_margin=768,
            vision=False,
            speculation=False,
        )
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        expected_lines = [
            f'set "LLAMA_ARG_HF_REPO={M.LAUNCHER_HF_REPO}"',
            f'set "LLAMA_ARG_HOST={M.LAUNCHER_HOST}"',
            f'set "LLAMA_ARG_PORT={M.LAUNCHER_PORT}"',
            'set "LLAMA_ARG_JINJA=1"',
            'set "LLAMA_ARG_N_PARALLEL=1"',
            'set "LLAMA_ARG_LOAD_MODE=mmap"',
            'set "LLAMA_ARG_LAZY_MODE=on"',
            'set "LLAMA_ARG_FLASH_ATTN=on"',
            'set "LLAMA_ARG_FIT=on"',
            'set "LLAMA_ARG_FIT_TARGET=768"',
            'set "LLAMA_ARG_CTX_SIZE=163840"',
            'set "LLAMA_ARG_BATCH=1024"',
            'set "LLAMA_ARG_UBATCH=512"',
            'set "LLAMA_ARG_CACHE_TYPE_K=f16"',
            'set "LLAMA_ARG_CACHE_TYPE_V=f16"',
            'set "LLAMA_ARG_THREADS=13"',
            'set "LLAMA_ARG_SPEC_TYPE=none"',
            'set "LLAMA_ARG_MMPROJ_AUTO=0"',
            'set "LLAMA_ARG_TOP_K=20"',
            'set "LLAMA_ARG_ENDPOINT_METRICS=1"',
        ]
        for line in expected_lines:
            assert line in text, f"line={line}"

    def test_flag_only_settings_never_get_an_environment_variable(self):
        """The six always-flag settings, plus speculation's three, never
        show up as an invented ``LLAMA_ARG_*`` name."""
        cfg = E.Config(speculation=True, repetitiveness=30, vision=True)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        for missing_env in (
            "LLAMA_ARG_THREADS_BATCH",
            "LLAMA_ARG_PRIO",
            "LLAMA_ARG_TEMP",
            "LLAMA_ARG_TOP_P",
            "LLAMA_ARG_MIN_P",
            "LLAMA_ARG_WARMUP",
            "LLAMA_ARG_NO_WARMUP",
            "LLAMA_ARG_SPEC_NGRAM_MOD_N_MATCH",
            "LLAMA_ARG_SPEC_NGRAM_MOD_N_MIN",
            "LLAMA_ARG_SPEC_NGRAM_MOD_N_MAX",
        ):
            assert missing_env not in text, f"missing_env={missing_env}"

    def test_environment_settings_never_show_up_as_a_bare_flag(self):
        cfg = E.Config(speculation=True, repetitiveness=30, vision=True)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        for missing_flag in (
            "--ctx-size",
            "--cache-type-k",
            "--cache-type-v",
            "--flash-attn",
            "--spec-type",
            "--fit-target",
            "--hf-repo",
            "--n-parallel",
        ):
            assert missing_flag not in text, f"missing_flag={missing_flag}"

    def test_image_input_off_has_no_mmproj_line_but_turns_auto_off(self):
        cfg = E.Config(vision=False)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert "LLAMA_ARG_MMPROJ=" not in text
        assert 'set "LLAMA_ARG_MMPROJ_AUTO=0"' in text

    def test_image_input_on_names_the_file_and_also_turns_auto_off(self):
        cfg = E.Config(vision=True)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert f'set "LLAMA_ARG_MMPROJ=%~dp0{M.LAUNCHER_MMPROJ_RELATIVE_PATH}"' in text
        assert 'set "LLAMA_ARG_MMPROJ_AUTO=0"' in text

    def test_no_mmproj_auto_flag_never_appears_in_either_branch(self):
        """The old flag-based way of disabling auto-detection is gone:
        ``LLAMA_ARG_MMPROJ_AUTO=0`` replaces it, on and off alike."""
        for vision in (False, True):
            cfg = E.Config(vision=vision)
            out = E.estimate(cfg)
            text = E.launcher_script(cfg, out)
            assert "--no-mmproj-auto" not in text, f"vision={vision}"
            assert 'set "LLAMA_ARG_MMPROJ_AUTO=0"' in text, f"vision={vision}"

    def test_speculation_off_writes_none_and_drops_the_three_flags(self):
        cfg = E.Config(speculation=False)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert 'set "LLAMA_ARG_SPEC_TYPE=none"' in text
        for flag in (
            "--spec-ngram-mod-n-match",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-max",
        ):
            assert flag not in text

    def test_speculation_on_writes_ngram_mod_and_keeps_the_three_flags(self):
        cfg = E.Config(speculation=True, repetitiveness=10)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert 'set "LLAMA_ARG_SPEC_TYPE=ngram-mod"' in text
        assert (
            "--spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 "
            "--spec-ngram-mod-n-max 24"
        ) in text

    def test_a_spilling_configuration_puts_the_warning_before_the_first_set_line(self):
        degraded = _grid_config(262144, 2048)  # the measured degraded run
        failing = E.Config(
            ctx=262144, ubatch=8192, fit_margin=1024, vram_total_mib=16376
        )
        for cfg in (degraded, failing):
            out = E.estimate(cfg)
            assert out.status in (E.DEGRADED, E.FAIL), f"status={out.status}"
            text = E.launcher_script(cfg, out)
            lines = text.splitlines()
            warning_index = next(i for i, line in enumerate(lines) if "WARNING" in line)
            first_set_index = next(
                i for i, line in enumerate(lines) if line.startswith('set "')
            )
            assert warning_index < first_set_index, f"status={out.status}"

    def test_a_fitting_configuration_has_no_warning_block(self):
        cfg = E.Config(ctx=163840, ubatch=512)
        out = E.estimate(cfg)
        assert out.status == E.OK
        assert "WARNING" not in E.launcher_script(cfg, out)

    def test_the_file_starts_with_echo_off_and_ends_with_the_exit_line(self):
        for cfg in (
            E.Config(),
            E.Config(vision=True, speculation=True),
            E.Config(ctx=262144, ubatch=8192, vram_total_mib=16376),
        ):
            out = E.estimate(cfg)
            text = E.launcher_script(cfg, out)
            assert text.startswith("@echo off"), f"cfg={cfg}"
            assert text.endswith("exit /b %ERRORLEVEL%"), f"cfg={cfg}"

    def test_a_clamped_micro_batch_says_so_and_uses_the_clamped_value(self):
        cfg = E.Config(ctx=4096, ubatch=8192)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert "clamped" in text.lower()
        assert 'set "LLAMA_ARG_UBATCH=4096"' in text
        assert 'set "LLAMA_ARG_UBATCH=8192"' not in text

    def test_a_different_machine_is_named_in_a_comment(self):
        cfg = E.Config(vram_total_mib=24576)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert "RTX 4090 Laptop" in text
        assert "RTX 4090 Laptop" not in E.launcher_script(
            E.Config(), E.estimate(E.Config())
        )

    def test_alias_and_chat_template_lines_appear_for_a_default_configuration(self):
        cfg = E.Config()
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        assert f'set "LLAMA_ARG_ALIAS={M.LAUNCHER_ALIAS}"' in text
        assert (
            'set "LLAMA_ARG_CHAT_TEMPLATE_FILE=%~dp0'
            f'{M.LAUNCHER_CHAT_TEMPLATE_RELATIVE_PATH}"'
        ) in text

    def test_alias_and_chat_template_lines_survive_every_branch(self):
        """Neither line depends on any slider: they belong in every
        generated launcher, image input, speculation and the spill to the
        processor alike."""
        configs = {
            "image on": E.Config(vision=True),
            "image off": E.Config(vision=False),
            "speculation on": E.Config(speculation=True, repetitiveness=10),
            "speculation off": E.Config(speculation=False),
            "spills to the processor": _grid_config(262144, 2048),
        }
        for label, cfg in configs.items():
            out = E.estimate(cfg)
            text = E.launcher_script(cfg, out)
            assert f'set "LLAMA_ARG_ALIAS={M.LAUNCHER_ALIAS}"' in text, (
                f"config={label}"
            )
            assert (
                'set "LLAMA_ARG_CHAT_TEMPLATE_FILE=%~dp0'
                f'{M.LAUNCHER_CHAT_TEMPLATE_RELATIVE_PATH}"'
            ) in text, f"config={label}"

    def test_alias_and_chat_template_sit_after_hf_repo_and_before_memory_settings(self):
        cfg = E.Config()
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        lines = text.splitlines()
        hf_repo_index = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_HF_REPO=')
        )
        alias_index = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_ALIAS=')
        )
        chat_template_index = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_CHAT_TEMPLATE_FILE=')
        )
        fit_target_index = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_FIT_TARGET=')
        )
        ctx_size_index = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_CTX_SIZE=')
        )
        assert hf_repo_index < alias_index
        assert alias_index < chat_template_index
        assert chat_template_index < fit_target_index
        assert chat_template_index < ctx_size_index

    def test_alias_and_chat_template_never_show_up_on_the_flag_line(self):
        cfg = E.Config(speculation=True, repetitiveness=30, vision=True)
        out = E.estimate(cfg)
        text = E.launcher_script(cfg, out)
        lines = text.splitlines()
        flag_start = next(
            i for i, line in enumerate(lines) if line == "llama-server.exe ^"
        )
        flag_end = next(i for i, line in enumerate(lines) if line.strip() == "%*")
        flag_block = "\n".join(lines[flag_start : flag_end + 1])
        assert "LLAMA_ARG_ALIAS" not in flag_block
        assert "LLAMA_ARG_CHAT_TEMPLATE_FILE" not in flag_block
        assert "--alias" not in flag_block
        assert "--chat-template-file" not in flag_block


class TestChatTemplateFetchBlock:
    """The launcher has to work on a machine that copied only the launcher.

    llama-server downloads the model itself, but the chat template is a
    local file, and the template inside the model file refuses a system
    message that is not the first one. Without these lines the person
    gets HTTP 500 on the first request from an agent harness.
    """

    def _text(self, **kwargs):
        cfg = E.Config(**kwargs)
        return E.launcher_script(cfg, E.estimate(cfg))

    def test_the_block_is_in_every_launcher(self):
        for kwargs in (
            {},
            {"ctx": 65536, "ubatch": 4096},
            {"vision": False},
            {"speculation": False},
            {"vram_total_mib": 12288},
        ):
            text = self._text(**kwargs)
            assert (
                'if exist "%LLAMA_ARG_CHAT_TEMPLATE_FILE%" goto template_ready' in text
            ), f"kwargs={kwargs}"
            assert "curl.exe -L -f -s -S -o" in text, f"kwargs={kwargs}"

    def test_the_url_follows_the_newest_revision(self):
        """The author keeps repairing the template, so a machine being set
        up should get the repairs. What was measured here is recorded in
        measured.py instead of being frozen into the launcher."""
        text = self._text()
        assert M.LAUNCHER_CHAT_TEMPLATE_FALLBACK_URL in text
        assert "/resolve/main/" in text
        assert M.LAUNCHER_CHAT_TEMPLATE_MEASURED_REVISION not in text

    def test_the_measured_revision_is_still_on_record(self):
        assert len(M.LAUNCHER_CHAT_TEMPLATE_MEASURED_REVISION) == 40
        assert M.LAUNCHER_CHAT_TEMPLATE_MEASURED_BYTES == 28234
        assert M.LAUNCHER_CHAT_TEMPLATE_MEASURED_VERSION.startswith("qwen3.8-")

    def test_the_launcher_prints_the_version_it_fetched(self):
        """A template that changes upstream has to be visible. The line
        reads the version out of the file it just wrote."""
        text = self._text()
        assert "do echo Template version: %%V" in text
        assert 'findstr /c:"template_version"' in text

    def test_the_block_uses_labels_and_opens_no_parenthesised_block(self):
        """Measured on this machine: cmd.exe reports 0 instead of 1 when
        ``exit /b 1`` sits inside two parenthesised blocks and another
        command follows the inner one. A wrapper would then treat a
        failed fetch as a good start. The for loop below is one command,
        not a block, so it does not bring the trap back."""
        text = self._text()
        block = text.split('if exist "%LLAMA_ARG_CHAT_TEMPLATE_FILE%"')[1]
        block = block.split(":template_ready")[0]
        for line in block.split("\n"):
            stripped = line.strip()
            assert not stripped.endswith("("), stripped
            assert stripped != ")"
        assert "if errorlevel 1 goto template_failed" in block
        assert ":template_failed" in block
        assert "exit /b 1" in block

    def test_the_block_runs_before_the_server_starts(self):
        text = self._text()
        lines = text.split("\n")
        guard = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('if exist "%LLAMA_ARG_CHAT_TEMPLATE_FILE%"')
        )
        template_set = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('set "LLAMA_ARG_CHAT_TEMPLATE_FILE=')
        )
        server = next(
            i for i, line in enumerate(lines) if line.startswith("llama-server.exe")
        )
        ready = next(i for i, line in enumerate(lines) if line == ":template_ready")
        assert template_set < guard
        assert guard < ready
        assert ready < server

    def test_the_fetch_never_reaches_the_flag_line(self):
        text = self._text()
        flag_block = text.split("llama-server.exe")[1]
        assert "curl" not in flag_block
        assert "template_ready" not in flag_block

    def test_the_failure_path_tells_the_person_where_to_get_the_file(self):
        text = self._text()
        failure = text.split(":template_failed")[1].split(":template_ready")[0]
        assert M.LAUNCHER_CHAT_TEMPLATE_FALLBACK_URL in failure
        assert "popd" in failure
        assert "exit /b 1" in failure


class TestMmprojFetchBlock:
    """Image input needs a projector file that llama-server never fetches
    for itself, because it is named as a file path and
    LLAMA_ARG_MMPROJ_AUTO is 0. A machine that copied only the launcher
    has to get it from the model's own repository, the same way it
    already gets the chat template.
    """

    def _text(self, **kwargs):
        cfg = E.Config(**kwargs)
        return E.launcher_script(cfg, E.estimate(cfg))

    def test_vision_on_carries_the_fetch_block(self):
        text = self._text(vision=True)
        assert M.LAUNCHER_MMPROJ_URL in text
        assert ":mmproj_ready" in text
        assert 'if exist "%LLAMA_ARG_MMPROJ%" goto mmproj_ready' in text

    def test_vision_off_carries_neither_the_url_nor_the_label(self):
        text = self._text(vision=False)
        assert M.LAUNCHER_MMPROJ_URL not in text
        assert "mmproj_ready" not in text
        assert "mmproj_failed" not in text

    def test_the_url_is_the_model_repository_without_the_quant_suffix(self):
        assert ":" not in M.LAUNCHER_MMPROJ_REPO
        assert M.LAUNCHER_HF_REPO.startswith(M.LAUNCHER_MMPROJ_REPO + ":")
        assert M.LAUNCHER_MMPROJ_REPO in M.LAUNCHER_MMPROJ_URL
        assert "/resolve/main/" in M.LAUNCHER_MMPROJ_URL

    def test_the_measured_size_is_still_on_record(self):
        assert M.LAUNCHER_MMPROJ_MEASURED_BYTES == 904003840

    def test_the_download_lands_on_a_temporary_name_before_the_rename(self):
        """A run interrupted mid-download must not leave a file at the
        final name that looks complete. curl writes to a ``.part`` name,
        and only a successful ``move`` puts it where LLAMA_ARG_MMPROJ
        points."""
        text = self._text(vision=True)
        block = text.split('if exist "%LLAMA_ARG_MMPROJ%" goto mmproj_ready')[1].split(
            ":mmproj_ready"
        )[0]
        curl_line = next(
            line for line in block.split("\n") if line.startswith("curl.exe")
        )
        assert '"%LLAMA_ARG_MMPROJ%.part"' in curl_line
        assert '-o "%LLAMA_ARG_MMPROJ%"' not in curl_line
        assert 'move /y "%LLAMA_ARG_MMPROJ%.part" "%LLAMA_ARG_MMPROJ%"' in text

    def test_the_block_uses_labels_and_opens_no_parenthesised_block(self):
        text = self._text(vision=True)
        block = text.split('if exist "%LLAMA_ARG_MMPROJ%"')[1].split(":mmproj_ready")[0]
        for line in block.split("\n"):
            stripped = line.strip()
            assert not stripped.endswith("("), stripped
            assert stripped != ")"
        assert "if errorlevel 1 goto mmproj_download_failed" in block
        assert "if errorlevel 1 goto mmproj_failed" in block
        assert ":mmproj_failed" in block

    def test_the_failure_path_exits_with_code_one(self):
        text = self._text(vision=True)
        failure = text.split(":mmproj_failed")[1].split(":mmproj_ready")[0]
        assert M.LAUNCHER_MMPROJ_URL in failure
        assert "%LLAMA_ARG_MMPROJ%" in failure
        assert "popd" in failure
        assert "exit /b 1" in failure

    def test_the_block_sits_after_the_chat_template_block_and_before_the_flag_line(
        self,
    ):
        text = self._text(vision=True)
        lines = text.splitlines()
        template_ready = next(
            i for i, line in enumerate(lines) if line == ":template_ready"
        )
        mmproj_guard = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('if exist "%LLAMA_ARG_MMPROJ%"')
        )
        mmproj_ready = next(
            i for i, line in enumerate(lines) if line == ":mmproj_ready"
        )
        server = next(
            i for i, line in enumerate(lines) if line.startswith("llama-server.exe")
        )
        assert template_ready < mmproj_guard
        assert mmproj_guard < mmproj_ready
        assert mmproj_ready < server

    def test_the_fetch_never_reaches_the_flag_line(self):
        text = self._text(vision=True)
        flag_block = text.split("llama-server.exe")[1]
        assert "curl" not in flag_block
        assert "mmproj_ready" not in flag_block
