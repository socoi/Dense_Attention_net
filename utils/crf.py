import numpy as np
import pydensecrf.densecrf as dcrf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_crf(img, output_probs):
    EPSILON = 1e-8
    img = np.transpose(img, (1, 2, 0))
    output_probs = np.transpose(output_probs, (1, 2, 0))

    h = output_probs.shape[0]
    w = output_probs.shape[1]

    # output_probs = np.expand_dims(output_probs, 0)
    # output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    n_energy = -np.log((1.0 - output_probs + EPSILON)) / (1.05 * sigmoid(1 - output_probs))
    p_energy = -np.log(output_probs + EPSILON) / (1.05 * sigmoid(output_probs))
    U = np.zeros((2, h * w))
    U = np.ascontiguousarray(U, dtype=float)
    img = np.ascontiguousarray(img)
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U.astype(np.float32))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=5, srgb=5, rgbim=img, compat=5)
    infer = np.array(d.inference(1))
    Q = infer[1, :]
    Q = Q.reshape((h, w))
    # Q = d.inference(1)
    # Q = np.argmax(np.array(Q), axis=0).reshape((h, w))



    # U = -np.log(output_probs)
    # U = U.reshape((2, -1))
    # U = np.ascontiguousarray(U, dtype=float)
    # img = np.ascontiguousarray(img)
    # d.setUnaryEnergy(U.astype(np.float32))
    #d.addPairwiseGaussian(sxy=20, compat=3)
    #d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    # Q = d.inference(5)
    # Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
