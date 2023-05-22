import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def divide():
    img = cv2.imread(r"static\uploads\DocScanner_27_May_2022_8-37_pm_page-0001.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 900, 600)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # Select the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    table_contour = contours[max_index]

    x, y, w, h = cv2.boundingRect(table_contour)

    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Select the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    table_contour = contours[max_index]

    # Create a mask
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [table_contour], -1, 255, -1)

    # Extract the table
    table = cv2.bitwise_and(img, img, mask=mask)

    # Define the points that define the region
    points = np.array([[8, 867], [5, 1252], [573, 1254], [584,874]])

    # Create a mask with the same size as the image
    mask = np.zeros(img_with_contours.shape[:2], dtype=np.uint8)

    # Draw the region defined by the points in the mask
    cv2.fillPoly(mask, [points], 255)

    # Apply the mask to the image to extract the region
    result = cv2.bitwise_and(img, img, mask=mask)



    # Define the points that form the reference rectangle
    ref_points = np.array([[8, 867], [5, 1252], [573, 1254], [584,874]], dtype=np.float32)

    # Calculate the width and height of the reference rectangle
    ref_width = np.abs(ref_points[0][0] - ref_points[2][0])
    ref_height = np.abs(ref_points[0][1] - ref_points[2][1])

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and find rectangles that have a similar size to the reference rectangle
    for contour in contours:
        # Check if the contour has enough points to form a rectangle
        if len(contour) >= 4:
            # Fit a rectangle to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Calculate the width and height of the rectangle
            width = np.abs(box[0][0] - box[2][0])
            height = np.abs(box[0][1] - box[2][1])

            # Check if the width and height of the rectangle are similar to the reference rectangle
            if abs(width - ref_width) < 20 and abs(height - ref_height) < 20:
                # Draw the rectangle on the image
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                # Get the coordinates of the rectangle
                x, y, w, h = cv2.boundingRect(box)
                print(x,y,w,h)
                # Crop the image to only include the rectangle
                crop_img = img[y:y+h, x:x+w]

    # Save the result
    # cv2.imwrite("output.jpg", crop_img)
    cv2.imshow("image :",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define the size of the cells
    cell_width = 466
    cell_height = 66

    # Define the size of the cells
    cell_width = 466
    cell_height = 66   # 6 is the number of rows

    # Create an empty list to store the cells
    cells = []

    # Divide the rectangle into cells
    for i in range(y+66, y + h, cell_height):
            cell = img[i:i + 66, 113: cell_width]
            cells.append(cell)

    # Save the cells
    for i, cell in enumerate(cells):
        cv2.imwrite(f"images/cell_{i}.jpg", cell)
        cv2.imshow("cells",cell)
        cv2.waitKey(0)
