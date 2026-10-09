'''
Reader for EDF+ files.
TODO:
 - add support for log-transformed channels:
   http://www.edfplus.info/specs/edffloat.html and test with
   data generated with
   http://www.edfplus.info/downloads/software/NeuroLoopGain.zip.
 - check annotations with Schalk's Physiobank data.
Copyright (c) 2012 Boris Reuderink.
'''
import functools
import re, datetime, operator, logging
import numpy as np
from collections import namedtuple

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)

class EDFEndOfData(Exception): pass


# def tal(tal_str):
#   '''Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
#   stream.
#   '''
#   exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
#     '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
#     '(\x14(?P<annotation>[^\x00]*))?' + \
#     '(?:\x14\x00)'
#
#   def annotation_to_list(annotation):
#       # 将bytes解码为str，再按\x14分割
#       return annotation.decode('utf-8').split('\x14') if annotation else []
#
#   def parse(dic):
#     return (
#       float(dic['onset']),
#       float(dic['duration']) if dic['duration'] else 0.,
#       annotation_to_list(dic['annotation']))
#
#   return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]
def tal(tal_str):
  '''Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  '''
  # 将正则表达式模式改为字节串
  exp = b'(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
        b'(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
        b'(\x14(?P<annotation>[^\x00]*))?' + \
        b'(?:\x14\x00)'

  def parse(dic):
    # 处理字节数据，解码为字符串后再转换为浮点数
    onset = float(dic['onset'].decode('latin-1'))
    duration = float(dic['duration'].decode('latin-1')) if dic['duration'] else 0.
    annotation = annotation_to_list(dic['annotation'])
    return (onset, duration, annotation)

  def annotation_to_list(annotation):
    if annotation:
      # 将字节解码为字符串，然后按字节分隔符 b'\x14' 分割
      # 注意：这里需要先解码再分割，或者按字节分割后再解码
      annotations = annotation.split(b'\x14')
      return [ann.decode('latin-1', errors='ignore') for ann in annotations]
    return []

  return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]

def edf_header(f):
  h = {}
  assert f.tell() == 0  # check file position
  assert f.read(8) == b'0       '   # 注意这里比较字节串

  # recording info)
  h['local_subject_id'] = f.read(80).decode('latin-1').strip()
  h['local_recording_id'] = f.read(80).decode('latin-1').strip()

  # parse timestamp
  date_str = f.read(8).decode('latin-1')
  time_str = f.read(8).decode('latin-1')
  # 使用正则表达式提取数字，注意日期和时间字符串的格式是固定的，但可能包含空格或其他非数字字符？
  # 根据EDF标准，日期格式是：dd.mm.yy，时间格式是：hh.mm.ss
  # 但是，有时可能使用空格作为分隔符？所以这里使用正则表达式提取数字
  date_numbers = re.findall('(\d+)', date_str)
  time_numbers = re.findall('(\d+)', time_str)
  if len(date_numbers) == 3 and len(time_numbers) == 3:
    (day, month, year) = [int(x) for x in date_numbers]
    (hour, minute, sec) = [int(x) for x in time_numbers]
    # 注意：年份可能是两位，EDF标准中年份是两位，但通常表示1984年之后的年份，所以加上2000
    if year < 100:
      year += 2000
    h['date_time'] = str(datetime.datetime(year, month, day, hour, minute, sec))
  else:
    h['date_time'] = None

  # misc
  header_nbytes = int(f.read(8).decode('latin-1'))
  subtype = f.read(44).decode('latin-1')[:5]
  h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
  h['contiguous'] = subtype != 'EDF+D'
  h['n_records'] = int(f.read(8).decode('latin-1'))
  h['record_length'] = float(f.read(8).decode('latin-1'))  # in seconds
  nchannels = h['n_channels'] = int(f.read(4).decode('latin-1'))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).decode('latin-1').strip() for n in channels]
  h['transducer_type'] = [f.read(80).decode('latin-1').strip() for n in channels]
  h['units'] = [f.read(8).decode('latin-1').strip() for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8).decode('latin-1')) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8).decode('latin-1')) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8).decode('latin-1')) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8).decode('latin-1')) for n in channels])
  h['prefiltering'] = [f.read(80).decode('latin-1').strip() for n in channels]
  h['n_samples_per_record'] = [int(f.read(8).decode('latin-1')) for n in channels]
  f.read(32 * nchannels)  # reserved

  assert f.tell() == header_nbytes
  return h

class BaseEDFReader:
  def __init__(self, file):
    self.file = file


  def read_header(self):
    self.header = h = edf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    assert np.all(phys_range > 0)
    assert np.all(dig_range > 0)
    self.gain = phys_range / dig_range


  def read_raw_record(self):
    '''Read a record with data and return a list containing arrays with raw
    bytes.
    '''
    result = []
    for nsamp in self.header['n_samples_per_record']:
      samples = self.file.read(nsamp * 2)
      if len(samples) != nsamp * 2:
        raise EDFEndOfData
      result.append(samples)
    return result


  def convert_record(self, raw_record):
    '''Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    '''
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
      if h['label'][i] == EVENT_CHANNEL:
        ann = tal(samples)
        time = ann[0][0]
        events.extend(ann[1:])
        # print(i, samples)
        # exit()
      else:
        # 2-byte little-endian integers
        dig = np.fromstring(samples, '<i2').astype(np.float32)
        phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
        signals.append(phys)

    return time, signals, events


  def read_record(self):
    return self.convert_record(self.read_raw_record())


  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    except EDFEndOfData:
      pass


def load_edf(edffile):
  '''Load an EDF+ file.
  Very basic reader for EDF and EDF+ files. While BaseEDFReader does support
  exotic features like non-homogeneous sample rates and loading only parts of
  the stream, load_edf expects a single fixed sample rate for all channels and
  tries to load the whole file.
  Parameters
  ----------
  edffile : file-like object or string
  Returns
  -------
  Named tuple with the fields:
    X : NumPy array with shape p by n.
      Raw recording of n samples in p dimensions.
    sample_rate : float
      The sample rate of the recording. Note that mixed sample-rates are not
      supported.
    sens_lab : list of length p with strings
      The labels of the sensors used to record X.
    time : NumPy array with length n
      The time offset in the recording for each sample.
    annotations : a list with tuples
      EDF+ annotations are stored in (start, duration, description) tuples.
      start : float
        Indicates the start of the event in seconds.
      duration : float
        Indicates the duration of the event in seconds.
      description : list with strings
        Contains (multiple?) descriptions of the annotation event.
  '''
  if isinstance(edffile, str):  # 用str替代basestring
      with open(edffile, 'rb') as f:
          return load_edf(f)  # convert filename to file

  reader = BaseEDFReader(edffile)
  reader.read_header()

  h = reader.header
  log.debug('EDF header: %s' % h)

  # get sample rate info
  nsamp = np.unique(
    [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
    if l != EVENT_CHANNEL])
  assert nsamp.size == 1, 'Multiple sample rates not supported!'
  sample_rate = float(nsamp[0]) / h['record_length']

  rectime, X, annotations = zip(*reader.records())
  X = np.hstack(X)
  annotations = functools.reduce(operator.add, annotations) if annotations else []
  chan_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]

  # create timestamps
  if reader.header['contiguous']:
    time = np.arange(X.shape[1]) / sample_rate
  else:
    reclen = reader.header['record_length']
    within_rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
    time = np.hstack([t + within_rec_time for t in rectime])

  tup = namedtuple('EDF', 'X sample_rate chan_lab time annotations')
  return tup(X, sample_rate, chan_lab, time, annotations)
