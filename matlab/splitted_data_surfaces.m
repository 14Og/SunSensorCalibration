az = readmatrix("matlab/data/azimuth.txt");
el = readmatrix("matlab/data/elevation.txt");
az_err = readmatrix("matlab/data/azimuth_error.txt");



x_az = az(:,1);
y_az = az(:,2);
z_az = az(:,3);

x_el = el(:,1);
y_el = el(:,2);
z_el = el(:,3);

x_az_er = az_err(:,1);
y_az_er = az_err(:,2);

% curveFitter(x_az,y_az,z_az);
% curveFitter(x_el,y_el,z_el);
plot(x_az_er, y_az_er, "o", "MarkerSize", 5);
% curveFitter(x3,y3,z3);
% curveFitter(x4,y4,z4);
% plot3(x2,y2,z2,'-o','Color','b','MarkerSize',10,...
%     'MarkerFaceColor','#D9FFFF');


