/// Two short originals so there is something to play straight away.
library;

/// A sample added before scripts had an index gets the new version, unless
/// it was edited since. Returns null when there is nothing to upgrade.
String? upgradedSample(String stored) {
  for (final s in [samplePodcast, sampleAudiobook]) {
    if (stored != s && stored == _withoutIndex(s)) return s;
  }
  return null;
}

String _withoutIndex(String s) {
  final t = s
      .replaceAll(RegExp(r'^(?:summary|topics):.*\n', multiLine: true), '')
      .replaceAll(RegExp(r'^index:\n(?:  .*\n)*', multiLine: true), '');
  return t.startsWith('---\n---\n') ? t.substring(8) : t;
}

const samplePodcast = '''---
title: Why Is the Sky Blue?
style: podcast
music: curious
summary: Why sunlight turns the sky blue, why it isn't violet, and why sunsets glow orange.
topics: science, physics, light
voices:
  HOST: Radio Host
  CO-HOST: Co-Host
  GUEST: The Professor
index:
  - Welcome | Meet the hosts and today's everyday question, with a guest who has spent a lifetime studying light.
  - Sunlight is a mixture | White light is every colour mixed together. The sky doesn't make blue; it sorts it.
  - The scattering trick | Air molecules scatter blue light about five times more than red, and that's why the sky isn't violet.
  - Sunsets | At sunset light crosses far more air, so the blue is scattered away and the reds are left.
---

HOST: Welcome to Curious Minds, the show where we take one everyday question and chase it all the way down. I'm your host.

CO-HOST: And I'm the one who keeps asking "but why?" Today's question sounds simple. Why is the sky blue?

HOST: It sounds simple. But the answer takes us from sunlight, to tiny molecules, to the reason sunsets turn orange. And to help, we have a guest who has spent a lifetime thinking about light.

GUEST: Thank you both. It's a wonderful question, because the sky is a giant experiment running over our heads every single day.

## Part 1: Sunlight is a mixture

CO-HOST: Okay, Professor. Start us off. Sunlight looks white to me.

GUEST: It does, and that's the first surprise. White light is really every colour mixed together. A prism, or a rainbow, simply pulls the mixture apart, so you can see the reds, the greens and the blues side by side.

HOST: So the blue is already in the sunlight. The sky doesn't make it.

GUEST: Exactly. The sky only sorts it.

## Part 2: The scattering trick

CO-HOST: Sorts it how?

GUEST: The air is full of tiny molecules of nitrogen and oxygen, far smaller than the waves of light. When light hits them, some of it bounces off in a new direction. We call that scattering. And here's the key. Short waves, the blue ones, are scattered far more than long red waves. Roughly five times more.

CO-HOST: Wait, really? Five times?

GUEST: Really. So as sunlight crosses the sky, blue light gets bounced around in every direction, and some of it reaches your eyes from wherever you look. That's why the whole dome of the sky glows blue, not just the patch around the sun.

[pause 1]

HOST: Then why isn't the sky violet? Violet waves are even shorter.

GUEST: A brilliant follow-up. Sunlight contains less violet to begin with, some of it is absorbed high in the atmosphere, and our eyes are simply less sensitive to it. Put those together, and blue wins.

## Part 3: Sunsets

CO-HOST: So what happens at sunset? Everything goes orange and pink.

GUEST: At sunset, the light travels through much more air to reach you. By the time it arrives, most of the blue has been scattered away along the path. What's left is the long-wave light, the oranges and reds. So a sunset is the same physics, just seen from the other end of the journey.

HOST: I love that. The blue sky and the red sunset are the same story.

[music: outro]

HOST: Let's recap. Sunlight is a mixture of colours. Tiny air molecules scatter blue light much more than red, so blue reaches us from every direction. And at sunset, the blue has been scattered away, leaving the reds.

CO-HOST: Next time on Curious Minds: why the ocean is blue. Spoiler, it's not just reflecting the sky.

HOST: Thanks for listening, and stay curious.
''';

const sampleAudiobook = '''---
summary: The rise and slow fading of the Library of Alexandria, and the idea that outlived it.
topics: history, books
index:
  - A City of Books | How Alexandria set out to gather every book in the world, and copied scrolls from passing ships.
  - What Was Lost | Not one great fire but a slow fading, and how the library's dream lives on in every search box.
---
# The Library That Remembered Everything

## Chapter One: A City of Books

More than two thousand years ago, on the sunny coast of Egypt, there stood a city built around an idea. The idea was simple and enormous. Gather every book in the world into one place.

The city was Alexandria, and its library became the most famous collection of knowledge the ancient world had ever seen. Ships that sailed into the harbour were searched, not for gold, but for scrolls. Any book found on board was carried to the library and copied by hand. The copy went back to the ship. The original stayed.

Scholars came from every corner of the Mediterranean. They measured the size of the Earth, catalogued the stars, studied medicine and mathematics, and argued about poetry late into the night.

## Chapter Two: What Was Lost

No one knows exactly how the library ended. It was not one dramatic fire, as the legends say, but a long, slow fading: budgets cut, scholars driven away, buildings damaged in wars, scrolls left to crumble.

And yet the idea survived. Every library, every encyclopedia, every search box you have ever typed into, carries a little of Alexandria's dream. To gather what we know, and to keep it, so that the next curious person can begin where we left off.
''';
