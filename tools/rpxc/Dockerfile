FROM sdthirlwall/raspberry-pi-cross-compiler

# setup basic rust deps

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_TOOLCHAIN=nightly

RUN set -eux; \
    apt-get update; \
    apt-get --no-install-recommends install -y \
        wget \
        ; \
    wget 'https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init'; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --default-toolchain $RUST_TOOLCHAIN; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version; \
    apt-get remove -y --auto-remove \
        wget \
        ; \
    rm -rf /var/lib/apt/lists/*; \
    rustup target add armv7-unknown-linux-gnueabihf;