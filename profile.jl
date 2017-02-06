include("katcu.jl")
# slightly different version compared to kat.jl, since I need to initialize it from the same cluster
include("kataf.jl")

# initialize clusters, not random to make it easier to verify the result
clusters = [
    RGB{Float32}(0, 0, 0), RGB{Float32}(1, 0, 0),
    RGB{Float32}(0, 1, 0), RGB{Float32}(0, 0, 1),
    RGB{Float32}(1, 1, 0), RGB{Float32}(0, 1, 1),
]
impath = homedir()*"/juliastuff/HurricaneKatrina/images/1.jpg"

# test julia
img = load(impath)
imgs = RGB{Float32}.(restrict(img)) # scale and convert image
imgvec = vec(imgs)

tj1 = @elapsed kmeans(imgvec, clusters)
tj2 = @elapsed kmeans(imgvec, clusters)
means, labels = kmeans(imgvec, clusters)
segmented2 = reshape(map(i-> means[i], labels), size(imgs))


ArrayFire.setBackend(AF_BACKEND_CPU);
img = loadImage(impath, color = true);
img = scale(img, 0.5, 0.5);
w, h, c = size(img)
imvec = reshape(img, w*h, 1, c);
taf1cpu = @elapsed kmeans(imvec, clusters, "i.jpg", img)
taf2cpu = @elapsed kmeans(imvec, clusters, "i.jpg", img)
gc()

ArrayFire.setBackend(AF_BACKEND_OPENCL);
img = loadImage(impath, color = true);
img = scale(img, 0.5, 0.5);
w, h, c = size(img)
imvec = reshape(img, w*h, 1, c);
taf1cl = @elapsed kmeans(imvec, clusters, "i.jpg", img)
taf2cl = @elapsed kmeans(imvec, clusters, "i.jpg", img)
gc()
afmeans2, aflabels2 = kmeans(imvec, clusters, "i.jpg", img)
means2, labels2 = Array(afmeans2), Array(aflabels2);
segmented = reshape(map(i-> RGB((means2[1, i, :] ./ 255)...), labels2), size(img)[1:2])

# fails since restrict results in a slightly different image compared to ArrayFire.scale
# when looking at the result visually they're indistinguishable
#@assert c == labels

println("AF CPU threaded: ", min(taf1cpu, taf2cpu))
println("AF GPU: ", min(taf1cl, taf2cl))
println("Julia CPU Single: ", min(tj1, tj2))
