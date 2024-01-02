import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import music21 as m21
from midiutil import MIDIFile
import json

def create_midi_file(notes, output_file='output.mid', tempo=120):
  midi = MIDIFile(1)
  midi.addTempo(0, 0, tempo)
  for start, end, note_num in notes:
    track = 0
    channel = 0  
    volume = 100
    midi.addNote(track, channel, note_num, start, end, volume)
  with open(output_file, 'wb') as midi_file:
    midi.writeFile(midi_file)

def play_midi_file(file_path='output.mid'):
  mf = m21.midi.MidiFile()
  mf.open(file_path)
  mf.read()
  mf.close()
  score = m21.midi.translate.midiFileToStream(mf)
  print(f'Duration: {score.highestTime} seconds')
  sp = m21.midi.realtime.StreamPlayer(score)
  sp.play()


# notes = []

# df = pd.read_csv('../musicnet/musicnet/train_labels/2242.csv')

# for row in df.itertuples():
#   notes.append((row.start_beat, row.end_beat, row.note))

# print(notes)
# create_midi_file(notes)

# play_midi_file('../musicnet/musicnet_midis/musicnet_midis/Bach/2186_vs6_1.mid')
play_midi_file()