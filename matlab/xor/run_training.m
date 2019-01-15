function run_training()
disp('start training')
if init_params.mehod == train_method.type_bp
    disp('train use bp algorithm')
    train_bp();
elseif init_params.mehod == train_method.type_bp
    disp('train use cpl algorithm')
    train_cpl();
end
disp('end training')