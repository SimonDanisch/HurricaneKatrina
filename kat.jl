using ArrayFire
using Images
#setBackend(AF_BACKEND_OPENCL)

function kmeans(in::AbstractArray, k::Integer, image::AbstractString, img::AbstractArray, iter::Integer = 10)

    n = size(in, 1)
    d = size(in, 3)


    #data = in * 0.0
    mini = minimum(in, 1)
    maxi = maximum(in, 1)

    #mini  = reshape(mini, 1, 1, d)
    #maxi  = reshape(maxi, 1, 1, d)
    mini  = constant(mini[1], 1, 1, d)
    maxi  = constant(maxi[1], 1, 1, d)
    data  = (in - mini) ./ maxi

    means = rand(AFArray{Float32}, 1, k, d)
    curr_clusters = constant(0, size(data, 1)) - 1
    prev_clusters = constant(0, size(data, 1)) - 1

    for i = 1:iter
        prev_clusters = curr_clusters
        curr_clusters = clusterize(data, means)
        num_changed = countnz(prev_clusters .!= curr_clusters)
        if num_changed < (n/1000) + 1
            break
        end
        means = new_means(data, curr_clusters, k)
        @show i
    end

    means = maxi .* means + mini
    clusters = prev_clusters

    means, clusters
end

function clusterize(data::AbstractArray, means::AbstractArray)
    dists = distance(data, means)
    val, idx = minidx(dists, 2)
    idx
end

function distance(data::AbstractArray, means::AbstractArray)
    n = size(data, 1)
    k = size(means, 2)
    data2 = repeat(data, outer = [1,k,1])
    means2 = repeat(means, outer = [n,1,1])
    sum(abs(data2 - means2), 3)
end

function new_means(data::AbstractArray, clusters, k::Integer)
    d = size(data, 3)
    means = constant(0.0f0, 1, k, d)
    clustersd = repeat(clusters, outer = [1, 1, d])
    for i = 1:size(means,2)
        means[:, i, :] = sum(data .* (clusters .== i), 1) ./ (sum(clusters .== i, 1) + Float32(1e-5))
    end
    means
end

function driver()
    for i = 1:73
        img = loadImage("$i.jpg", color = true)
        img = scale(img, 0.5, 0.5)
        println("Loaded $i.img")
        w = size(img, 1)
        h = size(img, 2)
        c = size(img, 3)
        vec = reshape(img, w*h, 1, c)
        meansfull, clustersfull = kmeans(vec, 6, "i.jpg", img)
        m = Array(meansfull)
        c = Array(clustersfull)
        out_img = reshape(m[:,c,:], size(img)...) / 255
        println("Finished $i.img")
        #saveImage("proc$i.jpg", out_img)
        gc()
    end
end
driver()
