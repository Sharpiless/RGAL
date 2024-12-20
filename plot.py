import matplotlib.pyplot as plt

nums = [8*200, 16*200, 32*200, 64*200, 128*200, 256*200]
x = [i+1 for i in range(len(nums))]
ours = [86.43, 88.41, 89.29, 90.13, 90.70, 91.11]

#128 90.70
# 64 90.13
# 32 89.29
# 16 88.41
# 8  86.43

l1 = plt.plot(x, ours, 'g--', label='ours')
# plt.plot(x, ours, 'gc--', x, ohs, 'bc--', x, triplets, 'gc--')
plt.plot(x, ours, 'gc--')
plt.xticks(x, nums)
plt.title('')
plt.xlabel('Train Epoches')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid()
plt.savefig('plot.png')
