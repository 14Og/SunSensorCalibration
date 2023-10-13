az = readmatrix("data/azimuth.txt");
el = readmatrix("data/elevation.txt");
az_error = readmatrix("data/azimuth_error.txt");


x_az = az(:,1);
y_az = az(:,2);
z_az = az(:,3);

x_el = el(:,1);
y_el = el(:,2);
z_el = el(:,3);

x_az_err = az_error(:,1);
y_az_err = az_error(:,2);



% curveFitter(x_az,y_az,z_az);
% curveFitter(x_el,y_el,z_el);

% figure;

% h(1) = subplot(1,2,1);
plot3(x_az,y_az,z_az,'-o','Color','b','MarkerSize',5,...
    'MarkerFaceColor','#D9FFFF');
title("Azimuth angle");
grid("on");
xlabel("x");
ylabel("y");
zlabel("az");

% figure;

% h(2) = subplot(1,2,2);
plot3(x_el,y_el,z_el,'-o','Color','b','MarkerSize',5,...
    'MarkerFaceColor','#D9FFFF');
title("Elevation angle")
grid("on");
xlabel("x");
ylabel("y");
zlabel("el");


figure;

plot(x_az_err, y_az_err, ".", "Color", [0.8 0 1], "Markersize", 7);
title("Azimuth by atan error");
xlabel("angle");
ylabel("error");
