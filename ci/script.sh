#!/usr/bin/env bash
set -ex

cargo=cargo
target_param=""
features_param=""
if [ ! -z "$TARGET" ]; then
    rustup target add "$TARGET"
    cargo install -v cross --force
    cargo="cross"
    target_param="--target $TARGET"
fi

if [ -n "$FEATURES" ]; then
    features_param="--features=$FEATURES"
fi

$cargo build -v $target_param $features_param

if [  "$TRAVIS_RUST_VERSION" = "1.23.0" ]; then
    # testing requires building dev-deps, which require a newer Rust.
    exit 0
fi

$cargo test -v $target_param $features_param

# for now, `cross bench` is broken https://github.com/rust-embedded/cross/issues/239
if [ "$cargo" != "cross" ]; then
    $cargo bench -v $target_param $features_param -- --test # don't actually record numbers
fi

$cargo doc -v $target_param $features_param

$cargo test -v --release $features_param

if [ ! -z "$COVERAGE" ]; then
    if [ ! -z "$TARGET" ]; then
        echo "cannot record coverage while cross compiling"
        exit 1
    fi

    cargo install -v cargo-travis || echo "cargo-travis already installed"
    cargo coverage -v -m coverage-reports $features_param --kcov-build-location "$PWD/target"
    bash <(curl -s https://codecov.io/bash) -c -X gcov -X coveragepy -s coverage-reports
fi
