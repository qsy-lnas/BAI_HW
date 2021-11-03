from PIL import Image
 
def transparence2white(img):
#     img=img.convert('RGBA')  # 此步骤是将图像转为灰度(RGBA表示4x8位像素，带透明度掩模的真彩色；CMYK为4x8位像素，分色等)，可以省略
    sp=img.size
    width=sp[0]
    height=sp[1]
    print(sp)
    for yh in range(height):
        for xw in range(width):
            dot=(xw,yh)
            color_d=img.getpixel(dot)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            if(color_d[3]==0):
                color_d=(255,255,255,255)
                img.putpixel(dot,color_d)  # 赋值的方法是通过putpixel
    return img
 
for i in range(13, 29), range(30, 43), range(45, 58), range(60, 73):

    img=Image.open(str(i) + '.png')
    img=transparence2white(img)
    # img.show()  # 显示图片
    img.save(str(i) + '.png')  # 保存图片