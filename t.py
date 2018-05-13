#!/usr/bin/python  
  
import tensorflow as tf  
import numpy as np 
 
#fc1/weights:0
#fc1/biases:0
#fc1/alphas:0

root_model = '/home/yinyuecheng/tf2caffe/Onet/Onet_01/ONet-22'
out_path = "/home/yinyuecheng/tf2caffe_v2/Onet_out/tf_out/Onet.prototxt" 
template_file = '48net.prototxt'

tfile = open(template_file,'r')
tflines = tfile.readlines()
tfile.close()



def insertLines(tag,lines2):
    #find
    temp_tag = '###[template]:' + tag
    index = -1
    print temp_tag
    for i in range(0, len(tflines)):
        #print i, tflines[i]
        if(temp_tag in tflines[i]): 
            index = i
            print i, tflines[i]
            break
    if(index>=0): tflines.insert(index+1,lines2)
            

with tf.Session() as sess:  
    new_saver = tf.train.import_meta_graph(root_model+'.meta')  
    for var in tf.trainable_variables():  
        print var.name  
        #insertLines(var.name,0)
    new_saver.restore(sess, root_model)  
    all_vars = tf.trainable_variables() 
    
    for v in all_vars:
        name = v.name
        
        lines2 = []
        lines2.append('#########==writing:'+name+'\n')
        
        v_4d = np.array(sess.run(v))
        if name == 'fc1/weights:0':
            v_4d = np.swapaxes(v_4d, 0, 1)
            v_4d = np.reshape(v_4d,(256,3,3,128))
            v_4d = np.swapaxes(v_4d, 1, 3) 
            v_4d = np.swapaxes(v_4d, 2, 3) 
            v_4d = np.reshape(v_4d,(256,1152))
	    
            vshape = v_4d.shape[:]
            v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1])
            lines2.append('  blobs {\n')
            for vv in v_1d:
                lines2.append('    data: %.8f' % vv) 
                lines2.append('\n')
            lines2.append('    shape {\n')
            for s in vshape:
                lines2.append('      dim: ' + str(s))#print dims
                lines2.append('\n')
            lines2.append('    }\n')
            lines2.append('  }\n')
        elif v_4d.ndim == 4:  
            #v_4d.shape [ H, W, I, O ]        
            v_4d = np.swapaxes(v_4d, 0, 2) # swap H, I  
            v_4d = np.swapaxes(v_4d, 1, 3) # swap W, O  
            v_4d = np.swapaxes(v_4d, 0, 1) # swap I, O  
            #v_4d.shape [ O, I, H, W ]  
            
            vshape = v_4d.shape[:]  
            v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1]*v_4d.shape[2]*v_4d.shape[3])  
            lines2.append('  blobs {\n')  
            for vv in v_1d:  
                lines2.append('    data: %.8f' % vv)  
                lines2.append('\n')  
            lines2.append('    shape {\n')  
            for s in vshape:  
                lines2.append('      dim: ' + str(s))#print dims  
                lines2.append('\n')  
            lines2.append('    }\n')  
            lines2.append('  }\n')  
        elif v_4d.ndim == 1 :#do not swap
            lines2.append('  blobs {\n')  
            for vv in v_4d:  
                lines2.append('    data: %.8f' % vv)  
                lines2.append('\n')  
            lines2.append('    shape {\n')  
            lines2.append('      dim: ' + str(v_4d.shape[0]))#print dims  
            lines2.append('\n')  
            lines2.append('    }\n')  
            lines2.append('  }\n')  
        elif v_4d.ndim == 2:
            v_4d = np.swapaxes(v_4d, 0, 1) # swap I, O
            vshape = v_4d.shape[:]
            v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1])
	    lines2.append('  blobs {\n')
            for vv in v_1d:
                lines2.append('    data: %.8f' % vv) 
                lines2.append('\n')
            lines2.append('    shape {\n')
            for s in vshape:
                lines2.append('      dim: ' + str(s))#print dims
                lines2.append('\n')
            lines2.append('    }\n')
            lines2.append('  }\n')
        insertLines(name,lines2)

#print tflines
f = open(out_path,'w')
for line in tflines:
    f.writelines(line)
f.close()      

        
        
