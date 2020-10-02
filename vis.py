import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

outDir = '/home/sci/riddhishb/Documents/git-repos/ImplicitPrior/comparators/outputs/betab_mnist/100000/'
ifoutput = False
implicitFlag = False
if(implicitFlag):
	qz = np.load(outDir+'qz_samples.npy')
	qz0 = np.load(outDir+'qz0_samples.npy')
	hz = np.load(outDir+'high_pos_z.npy')
	hz0 = np.load(outDir+'high_pos_z0.npy')
	
	# genPts = np.load(outDir + 'generated_data.npy')
	lz = np.load(outDir+'low_pos_z.npy')
	lz0 = np.load(outDir+'low_pos_z0.npy')
	if ifoutput:
		lx = np.load(outDir+'low_pos_x.npy')
		hx = np.load(outDir+'high_pos_x.npy')
		origPts = np.load(outDir + 'original.npy')
		fig, ax = plt.subplots()
		ax.scatter(origPts[:, 0], origPts[:, 1], c='gray', label='Original-Data')
		# ax.scatter(hx[:5, 0], hx[:5, 1], c='red', marker='^', label='high-posterior')
		ax.scatter(lx[:5, 0], lx[:5, 1], c='black', marker='*', label='low-posterior')
		ax.legend()
		plt.savefig(outDir + 'data-space.png')
		plt.clf()
	# in z0 space
	fig, ax = plt.subplots()
	sns.kdeplot(qz0[:, 0], qz0[:, 1], shade=True, cmap='Blues', bw=0.3)
	ax.scatter(hz0[:2, 0], hz0[:2, 1], c='red', marker='^', label='high-posterior')
	ax.scatter(lz0[:2, 0], lz0[:2, 1], c='black', marker='*', label='low-posterior')
	ax.legend()
	plt.savefig(outDir + 'qz0.png')
	plt.clf()

	fig, ax = plt.subplots()
	sns.kdeplot(qz[:, 0], qz[:, 1], shade=True, cmap='Blues', bw=0.3)
	ax.scatter(hz[:2, 0], hz[:2, 1], c='red', marker='^', label='high-posterior')
	ax.scatter(lz[:2, 0], lz[:2, 1], c='black', marker='*', label='low-posterior')
	ax.legend()
	plt.xlim([-3, 7])
	plt.ylim([-7, 7])
	plt.savefig(outDir + 'qz.png')
	plt.clf()

	

else:

	qz = np.load(outDir+'qz_samples.npy')
	# qz0 = np.load(outDir+'qz0-values.npy')
	hz = np.load(outDir+'high_pos_z.npy')
	# hz0 = np.load(outDir+'highz0-values.npy')
	
	# genPts = np.load(outDir + 'generated_data.npy')
	lz = np.load(outDir+'low_pos_z.npy')
	# lz0 = np.load(outDir+'lowz0-values.npy')
	

	# in z0 space
	# fig, ax = plt.subplots()
	# sns.kdeplot(qz0[:, 0], qz0[:, 1], shade=True)
	# ax.scatter(hz0[:25, 0], hz0[:25, 1], c='red', marker='^', label='high-posterior')
	# ax.scatter(lz0[:25, 0], lz0[:25, 1], c='black', marker='*', label='low-posterior')
	# ax.legend()
	# plt.savefig(outDir + 'qz0.png')
	# plt.clf()

	fig, ax = plt.subplots()
	sns.kdeplot(qz[:, 0], qz[:, 1], shade=True, cmap='Blues', bw=0.3, gridsize=200)
	ax.scatter(hz[:5, 0], hz[:5, 1], c='red', marker='^', label='high-posterior')
	ax.scatter(lz[:5, 0], lz[:5, 1], c='black', marker='*', label='low-posterior')
	ax.legend()
	plt.savefig(outDir + 'qz.png')
	plt.clf()
	if ifoutput:
		lx = np.load(outDir+'low_pos_x.npy')
		hx = np.load(outDir+'high_pos_x.npy')
		origPts = np.load(outDir + 'original.npy')
		fig, ax = plt.subplots()
		ax.scatter(origPts[:, 0], origPts[:, 1], c='gray', label='Original-Data')
		ax.scatter(hx[:5, 0], hx[:5, 1], c='red', marker='^', label='high-posterior')
		ax.scatter(lx[:5, 0], lx[:5, 1], c='black', marker='*', label='low-posterior')
		ax.legend()
		plt.savefig(outDir + 'data-space.png')
		plt.clf()