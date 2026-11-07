# Local entry points. Every target runs at nice 10 and waits while NUC load is >= 20 (see scripts/versions.sh).
.PHONY: test-matrix test-matrix-smoke test-python build-all

test-matrix:        ## Python + mod build + Java tests + mod<->server interop, on every supported Minecraft version
	scripts/test_matrix.sh

test-matrix-smoke:  ## the above, plus a real Fabric server boot per version (needs MCROBOT_EULA_FILE)
	scripts/test_matrix.sh --smoke

test-python:
	PYTHONPATH=server nice -n 10 .venv/bin/python -m pytest server/tests -q

build-all:          ## mod jars for every version into mod/dist/
	JAVA_HOME=$${MCROBOT_JDK_25:-$$HOME/opt/jdk-25.0.5+8} nice -n 10 mod/gradlew -p mod --max-workers=2 buildAll
