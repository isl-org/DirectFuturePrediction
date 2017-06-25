import omg
import sys
import random
import os

def change_textures(in_map, textures_file):
    with open(os.path.join('/home/adosovit/work/future_pred/my_scenarios/paper', textures_file)) as f:
        textures = f.read().split()
        
    #print textures
    
    # three options are: 
    #   all random; 
    #   all walls the same, ceil the same, floor the same; 
    #   all the same
    probs = [1.,1.,1.]
    r = random.uniform(0,probs[0] + probs[1] + probs[2])
    if r < probs[0]:
        mode = 'all_rand'
    elif r < probs[0] + probs[1]:
        mode = 'walls_ceil_floor'
    else:
        mode = 'all_the_same'   
        
    print mode, textures_file
        
    map_editor = omg.MapEditor(in_map)
    
    apply_to_mid = False  # this has to be False for lab22
    if mode == 'all_rand':
        for s in map_editor.sidedefs:
            s.tx_up = random.choice(textures)
            s.tx_low = random.choice(textures)
            if apply_to_mid:
                s.tx_mid = random.choice(textures)
        for s in map_editor.sectors:
            s.tx_floor = random.choice(textures)
            s.tx_ceil = random.choice(textures)
    elif mode == 'walls_ceil_floor':
        wall_tx = random.choice(textures)
        floor_tx = random.choice(textures)
        ceil_tx = random.choice(textures)
        for s in map_editor.sidedefs:
            s.tx_up = wall_tx
            s.tx_low = wall_tx
            if apply_to_mid:
                s.tx_mid = wall_tx
        for s in map_editor.sectors:
            s.tx_floor = floor_tx
            s.tx_ceil = ceil_tx
    elif mode == 'all_the_same':
        all_tx = random.choice(textures)
        for s in map_editor.sidedefs:
            s.tx_up = all_tx
            s.tx_low = all_tx
            if apply_to_mid:
                s.tx_mid = all_tx
        for s in map_editor.sectors:
            s.tx_floor = all_tx
            s.tx_ceil = all_tx  
    else:
        raise Exception('Unknown mode', mode)
        
    out_map = map_editor.to_lumps()
    
    to_copy = ['BEHAVIOR']
    for t in to_copy:
        if t in in_map:
            out_map[t] = in_map[t]
    
    return out_map

if (len(sys.argv) != 3):
    print "    Usage:"
    print "    apply_random_textures.py input train/test/mix \n"
    raise Exception()

map_name = sys.argv[1]
mode = sys.argv[2]

in_file = '/home/adosovit/work/future_pred/my_scenarios/paper/%s.wad' % map_name
out_file = '/home/adosovit/work/future_pred/my_scenarios/paper/%s_manymaps_%s.wad' % (map_name, mode)


if mode == 'all':
    texture_files = ['all_textures.txt'] * 98
elif mode == 'train':
    texture_files = ['train_textures.txt'] * 98
elif mode == 'test':
    texture_files = ['test_textures.txt'] * 98
elif mode == 'mix':
    texture_files = ['train_textures.txt'] * 88 + ['test_textures.txt'] * 10
else:
    raise Exception('Unknown mode', mode)

wad = omg.WAD(in_file) 

for (nm,texturef) in enumerate(texture_files):
    wad.maps['MAP%.2d' % (nm+2)] = change_textures(wad.maps['MAP01'], texturef)

wad.to_file(out_file)

#for a in sys.argv[1:]:
    #if a == "-o":
        #break
    #print "Adding %s..." % a
    #w += omg.WAD(a)
#outpath = "merged.wad"
#if "-o" in sys.argv: outpath = sys.argv[-1]
#w.to_file(outpath)
