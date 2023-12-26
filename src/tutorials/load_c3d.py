from ezc3d import c3d

file_path = "/mnt/d/Dataset/Mocap/sub09/121301/Data 2022-12-13 17-18-58.c3d"

c = c3d(file_path)
print(c['parameters']['POINT']['LABELS'])