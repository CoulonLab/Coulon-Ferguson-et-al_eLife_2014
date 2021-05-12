
class objFromDict:
  "Usage: objFromDict(**dictionary).\nPrint gives the list of attributes."
  def __init__(self, **entries):
    self.__dict__.update(entries)
    for a in self.__dict__.keys():
      self.__dict__[a.replace('-','_').replace('*','_').replace('+','_').replace('/','_')]=self.__dict__.pop(a)
  def __repr__(self): return '** objFromDict attr. --> '+', '.join(filter((lambda s: (s[:2]+s[-2:])!='____'),dir(self)))



