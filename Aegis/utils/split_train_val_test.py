"""PHEME split entry point: recover official identities, group, then freeze.

Explicit inputs and a new output directory are required; run --help.
The old row-random splitter is retired because it could leak duplicate text.
"""
if __package__:
    from .freeze_pheme import main
else:
    from freeze_pheme import main


if __name__ == '__main__':
    main()
