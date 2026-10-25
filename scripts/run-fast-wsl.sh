#!/usr/bin/env bash
set -euo pipefail
mvn -q -DskipTests package
java -Xmx3g -jar target/kion-rl-java-1.0.0.jar "$@"
