path=''
out_path=""
inter_path=""
def crop(path):
    import cv2
    import matplotlib.pyplot as plt
    from skimage import measure, morphology
    from skimage.color import label2rgb
    from skimage.measure import regionprops
    from PIL import Image, ImageDraw
    import numpy as np
    from scipy.ndimage.filters import rank_filter
    
    img = cv2.imread(path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0

    
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1

        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)

    a4_constant = ((average/84.0)*250.0)+100

    b = morphology.remove_small_objects(blobs_labels, a4_constant)

    plt.imsave('pre_version.png', b)

    img = cv2.imread('pre_version.png', 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite(inter_path, img)

    def dilate(ary, N, iterations):

        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[(N-1)//2,:] = 1

        dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[:,(N-1)//2] = 1
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
        return dilated_image


    def props_for_contours(contours, ary):
        c_info = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            c_im = np.zeros(ary.shape)
            cv2.drawContours(c_im, [c], 0, 255, -1)
            c_info.append({
                'x1': x,
                'y1': y,
                'x2': x + w - 1,
                'y2': y + h - 1,
                'sum': np.sum(ary * (c_im > 0))/255
            })
        return c_info


    def union_crops(crop1, crop2):
        x11, y11, x21, y21 = crop1
        x12, y12, x22, y22 = crop2
        return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


    def intersect_crops(crop1, crop2):
        x11, y11, x21, y21 = crop1
        x12, y12, x22, y22 = crop2
        return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


    def crop_area(crop):
        x1, y1, x2, y2 = crop
        return max(0, x2 - x1) * max(0, y2 - y1)


    def find_border_components(contours, ary):
        borders = []
        area = ary.shape[0] * ary.shape[1]
        for i, c in enumerate(contours):
            x,y,w,h = cv2.boundingRect(c)
            if w * h > 0.5 * area:
                borders.append((i, x, y, x + w - 1, y + h - 1))
        return borders


    def angle_from_right(deg):
        return min(deg % 90, 90 - (deg % 90))


    def remove_border(contour, ary):
        c_im = np.zeros(ary.shape)
        r = cv2.minAreaRect(contour)
        degs = r[2]
        if angle_from_right(degs) <= 10.0:
            box = cv2.boxPoints(r)
            box = np.int0(box)
            cv2.drawContours(c_im, [box], 0, 255, -1)
            cv2.drawContours(c_im, [box], 0, 0, 4)
        else:
            x1, y1, x2, y2 = cv2.boundingRect(contour)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

        return np.minimum(c_im, ary)


    def find_components(edges, max_components=16):

        count = 21
        dilation = 5
        n = 1
        while count > 16:
            n += 1
            dilated_image = dilate(edges, N=3, iterations=n)
            dilated_image = np.uint8(dilated_image)
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
        return contours


    def find_optimal_components_subset(contours, edges):
        c_info = props_for_contours(contours, edges)
        c_info.sort(key=lambda x: -x['sum'])
        total = np.sum(edges) / 255
        area = edges.shape[0] * edges.shape[1]

        c = c_info[0]
        del c_info[0]
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        crop = this_crop
        covered_sum = c['sum']

        while covered_sum < total:
            changed = False
            recall = 1.0 * covered_sum / total
            prec = 1 - 1.0 * crop_area(crop) / area
            f1 = 2 * (prec * recall / (prec + recall))
            for i, c in enumerate(c_info):
                this_crop = c['x1'], c['y1'], c['x2'], c['y2']
                new_crop = union_crops(crop, this_crop)
                new_sum = covered_sum + c['sum']
                new_recall = 1.0 * new_sum / total
                new_prec = 1 - 1.0 * crop_area(new_crop) / area
                new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

                remaining_frac = c['sum'] / (total - covered_sum)
                new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
                if new_f1 > f1 or (
                        remaining_frac > 0.25 and new_area_frac < 0.15):
                    crop = new_crop
                    covered_sum = new_sum
                    del c_info[i]
                    changed = True
                    break

            if not changed:
                break

        return crop


    def pad_crop(crop, contours, edges, border_contour, pad_px=15):
        bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
        if border_contour is not None and len(border_contour) > 0:
            c = props_for_contours([border_contour], edges)[0]
            bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

        def crop_in_border(crop):
            x1, y1, x2, y2 = crop
            x1 = max(x1 - pad_px, bx1)
            y1 = max(y1 - pad_px, by1)
            x2 = min(x2 + pad_px, bx2)
            y2 = min(y2 + pad_px, by2)
            return crop

        crop = crop_in_border(crop)

        c_info = props_for_contours(contours, edges)
        changed = False
        for c in c_info:
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            this_area = crop_area(this_crop)
            int_area = crop_area(intersect_crops(crop, this_crop))
            new_crop = crop_in_border(union_crops(crop, this_crop))
            if 0 < int_area < this_area and crop != new_crop:
                print('%s -> %s' % (str(crop), str(new_crop)))
                changed = True
                crop = new_crop

        if changed:
            return pad_crop(crop, contours, edges, border_contour, pad_px)
        else:
            return crop


    def downscale_image(im, max_dim=2048):
        a, b = im.size
        if max(a, b) <= max_dim:
            return 1.0, im

        scale = 1.0 * max_dim / max(a, b)
        new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
        return scale, new_im


    def process_image(path, out_path):

        orig_im = Image.open(inter_path)
        scale, im = downscale_image(orig_im)

        edges = cv2.Canny(np.asarray(im), 100, 200)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        borders = find_border_components(contours, edges)
        borders.sort(key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

        border_contour = None
        if len(borders):
            border_contour = contours[borders[0][0]]
            edges = remove_border(border_contour, edges)

        edges = 255 * (edges > 0).astype(np.uint8)

        maxed_rows = rank_filter(edges, -4, size=(1, 20))
        maxed_cols = rank_filter(edges, -4, size=(20, 1))
        debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
        edges = debordered

        contours = find_components(edges)
        if len(contours) == 0:
            return



        crop = find_optimal_components_subset(contours, edges)
        crop = pad_crop(crop, contours, edges, border_contour)

        crop = [int(x / scale) for x in crop]  
        text_im = orig_im.crop(crop)
        text_im.save(out_path)
    process_image(path,out_path)
