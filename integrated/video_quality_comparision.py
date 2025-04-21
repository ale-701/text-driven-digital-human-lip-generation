if __name__ == "__main__":
    import os
    from PIL import Image

    # Folder paths
    folder1 = 'path_to_video1_frames'
    folder2 = 'path_to_video2_frames'

    # Collecting frame file names
    frames1 = sorted([f for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([f for f in os.listdir(folder2) if f.endswith('.png')])

    # Check for same number of frames
    if len(frames1) != len(frames2):
        raise ValueError("Different number of frames in folders")

    # Processing frames
    psnrs, ssims = [], []
    for f1, f2 in zip(frames1, frames2):
        img1 = np.asarray(Image.open(os.path.join(folder1, f1)))
        img2 = np.asarray(Image.open(os.path.join(folder2, f2)))

        mse, psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)

        psnrs.append(psnr)
        ssims.append(ssim)
        print(f'Frame: {f1}, mse = {mse:.6f}, psnr = {psnr:.6f}, ssim = {ssim:.6f}')

    # Average results
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    print(f'Average PSNR = {avg_psnr:.6f}, Average SSIM = {avg_ssim:.6f}')

