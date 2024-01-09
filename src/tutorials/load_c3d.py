from ezc3d import c3d

file_path = "/mnt/d/Dataset/Mocap/sub09/121301/Data 2022-12-13 17-18-58.c3d"
file_path_origin = "/mnt/d/Code/SOMA/support_files/evaluation_mocaps/original/SOMA_manual_labeled/soma_subject1/clap_001.c3d"

c = c3d(file_path)
c_origin = c3d(file_path_origin)

print(c.keys())
print(c['data']['point'].shape)
# print(c['data']['analogs'])
# print(c_origin['data'].keys())
# print(c_origin['data']['analogs'].shape)
# print(c_origin['data']['analogs'])