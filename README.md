## ResNet for Time Series Classification

ResNet for music(**MIDI** files) classification.

MIDI files are processed into sequential data using [note-based representation](https://salu133445.github.io/muspy/representations/index.html). Each note is represented as a  (time, pitch, duration, velocity) tuple, which is used in ResNet as 4 channels.

See [1](https://arxiv.org/abs/1809.04356) | [2](https://arxiv.org/abs/1611.06455) for details about using ResNet in TSC tasks.