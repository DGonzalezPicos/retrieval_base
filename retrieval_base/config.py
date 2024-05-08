import numpy as np
import pathlib

import json
import os

class Config:
    
    
    def __init__(self, 
                 path='/home/dario/phd/retrieval_base/',
                 target='TWA28',
                 run='testing_1',
                 ):
        self.path = pathlib.Path(path)
        self.target = target
        
        self.run = run
    
        if self.target is None:
            self.data_path = self.path / 'retrieval_outputs' / self.run / 'test_data'
        else:
            assert self.target not in self.path.parts, f'{self.target} not in {self.path.parts}'
            self.data_path = self.path / self.target / 'retrieval_outputs' / self.run / 'test_data'
            
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        
        # self.full_prefix = self.path / self.prefix
    
    def __call__(self, config_filename, config_path=None):
        
        
        self.config_filename = self.data_path / config_filename
        if config_path is not None:
            self.config_filename = pathlib.Path(config_path) / config_filename
        self.conf = json.load(open(self.config_filename))
        # set attributes
        for key, value in self.conf.items():
            setattr(self, key, value)
            
        return self
        
    def __str__(self):
        out = '** Config **\n'
        out += '-------------\n'
        
        if hasattr(self, 'conf'):
            for key, value in self.conf.items():
                out += f'{key} = {value}\n'
                
        return out
        
    def __repr__(self):
        return self.__str__()
    
    
        
    def save_json(self, filename, globals):
        outfile = self.data_path /  filename.replace('.py', '.txt')
        json_dump = dict()
        
        for key in list(globals.keys()):
            if key.startswith('__'):
                continue
            # print(type(globals()[key]))
            # if it is a module skip
            if isinstance(globals[key], type(os)):
                continue
            # if it is a function skip
            if callable(globals[key]):
                continue
            # if it is a posix path, save as str
            if isinstance(globals[key], type(pathlib.Path())):
                # json_dump[key] = str(globals()[key])
                continue
            # if isinstance(globals[key], [np.ndarray, list]):
            #     json_dump[key] = list(globals[key])
            #     continue
            
            print(f'{key} = {globals[key]}')
            
            json_dump[key] = globals[key]
        # remove key of json_dump (leads to circular reference)
        # json_dump.pop('json_dump')
            
        with open(outfile, 'w') as file:
            file.write(json.dumps(json_dump))
        print(f'Wrote {outfile}')
        return None
        
if __name__ == '__main__':
    
        
    path = pathlib.Path('/home/dario/phd/retrieval_base/')
    target = 'TWA28'
    run = 'rev_2'

    config_file = 'config_freechem.txt'
    
    conf = Config(path=path, target=target, run=run)
    conf(config_file)
    


