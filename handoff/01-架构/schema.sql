-- =============================================================
-- 鸡舍巡检数字孪生系统 · 数据库建表脚本 (schema.sql)
-- 对应《毕业生项目系统等成果归集要求》2.5.4.1
-- DBMS: MySQL 8.x | 字符集: utf8mb4 | 排序规则: utf8mb4_unicode_ci | 引擎: InnoDB
-- 说明: 字段以项目代码/文档真实出现的列为准；标注「需与后端实际库核对」的列
--       表示生产库可能存在不同定义，导入生产环境前请先核对。
-- =============================================================

CREATE DATABASE IF NOT EXISTS poultry_twin
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE poultry_twin;

-- -------------------------------------------------------------
-- 1. sys_dict_data 系统字典表
-- 用途: 下拉选项与状态文案（启用/停用、性别、显示/隐藏）
-- -------------------------------------------------------------
DROP TABLE IF EXISTS sys_dict_data;
CREATE TABLE sys_dict_data (
  dict_code   BIGINT       NOT NULL AUTO_INCREMENT COMMENT '字典编码',
  dict_label  VARCHAR(64)  NOT NULL                COMMENT '字典标签',
  dict_value  VARCHAR(64)  NOT NULL                COMMENT '字典键值',
  dict_type   VARCHAR(128) NOT NULL                COMMENT '字典类型',
  status      CHAR(1)      NOT NULL DEFAULT '0'    COMMENT '状态(0正常 1停用)',
  PRIMARY KEY (dict_code),
  KEY idx_dict_type (dict_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统字典表';

-- -------------------------------------------------------------
-- 2. sys_house 栋舍表
-- 用途: 栋舍台账，作为巡检/监测数据的归属维度
-- 注意: house_code 实测返回字符串，故用 VARCHAR
-- -------------------------------------------------------------
DROP TABLE IF EXISTS sys_house;
CREATE TABLE sys_house (
  house_code  VARCHAR(16)  NOT NULL                COMMENT '栋舍编号',
  house_name  VARCHAR(64)  DEFAULT NULL            COMMENT '栋舍名称',
  capacity    INT          DEFAULT 0               COMMENT '存栏量',
  status      CHAR(1)      DEFAULT '0'             COMMENT '状态(0正常 1停用)',
  tenant_id   VARCHAR(20)  NOT NULL DEFAULT '000000' COMMENT '租户ID',
  create_time DATETIME     DEFAULT NULL            COMMENT '创建时间',
  PRIMARY KEY (house_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='栋舍表';

-- -------------------------------------------------------------
-- 3. robot_info 机器人绑定表
-- 用途: 机器人 IMEI → 栋舍绑定，robotId→houseCode 链路源头
-- -------------------------------------------------------------
DROP TABLE IF EXISTS robot_info;
CREATE TABLE robot_info (
  robot_id    VARCHAR(32)  NOT NULL                COMMENT '机器人IMEI',
  robot_name  VARCHAR(64)  DEFAULT NULL            COMMENT '机器人名称',
  house_code  VARCHAR(16)  DEFAULT NULL            COMMENT '绑定栋舍编号',
  online      TINYINT      DEFAULT 0               COMMENT '在线状态(0离线 1在线)',
  bind_time   DATETIME     DEFAULT NULL            COMMENT '绑定时间',
  tenant_id   VARCHAR(20)  NOT NULL DEFAULT '000000' COMMENT '租户ID',
  PRIMARY KEY (robot_id),
  UNIQUE KEY uk_tenant_robot (tenant_id, robot_id),
  KEY idx_house (house_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='机器人绑定表';

-- -------------------------------------------------------------
-- 4. inspection_exception 巡检异常表 (核心事实表)
-- 用途: 记录巡检异常鸡，支持 afterId 增量拉取与三维落位
-- 说明: 生产库存在多套字段命名版本，本表以后端 RECORD_COLUMNS 真实列名为准。
--       exception_pos_x/y 与 judgment_value 实测为字符串，schema 保留 decimal/字符串
--       两种兼容形态的说明: 以下采用「类型宽松」定义以贴合实测，应用层仍需容错。
-- -------------------------------------------------------------
DROP TABLE IF EXISTS inspection_exception;
CREATE TABLE inspection_exception (
  inspection_exception_id BIGINT       NOT NULL AUTO_INCREMENT COMMENT '异常记录ID(增量游标)',
  robot_imei              VARCHAR(32)  DEFAULT NULL            COMMENT '机器人IMEI [需与后端实际库核对]',
  house_code              VARCHAR(16)  NOT NULL                COMMENT '栋舍编号',
  exception_type_id       INT          DEFAULT NULL            COMMENT '异常类型ID(1死鸡 4声音 5张嘴 6绝产)',
  exception_type_name     VARCHAR(64)  DEFAULT NULL            COMMENT '异常类型名称 [需与后端实际库核对]',
  exception_pos_x         VARCHAR(32)  DEFAULT NULL            COMMENT 'X坐标; 文档为decimal, 实测字符串(如11.630)',
  exception_pos_y         VARCHAR(32)  DEFAULT NULL            COMMENT 'Y坐标; 文档为decimal, 实测字符串(如5.800)',
  row_no                  INT          DEFAULT NULL            COMMENT '排号; 实测可能为-2(无效)',
  col_no                  INT          DEFAULT NULL            COMMENT '列号; 实测可能为NULL',
  tier_no                 INT          DEFAULT NULL            COMMENT '层号',
  exception_file_name     VARCHAR(255) DEFAULT NULL            COMMENT '异常图片文件名',
  exception_file_path     VARCHAR(255) DEFAULT NULL            COMMENT '异常图片路径前缀 [需与后端实际库核对]',
  exception_judgment_value VARCHAR(32) DEFAULT NULL            COMMENT '置信度; 实测为字符串或NULL',
  sync_time               DATETIME     DEFAULT NULL            COMMENT '同步时间',
  tenant_id               VARCHAR(20)  NOT NULL DEFAULT '000000' COMMENT '租户ID',
  PRIMARY KEY (inspection_exception_id),
  -- 核心复合索引: by-robot 查询 WHERE house_code=? AND id>? ORDER BY id LIMIT ?
  KEY idx_inspection_house_id (tenant_id, house_code, inspection_exception_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='巡检异常表';

-- -------------------------------------------------------------
-- 5. monitor_point_latest 监测点最新值表
-- 用途: 四个固定点位的最新环境值，UPSERT 写入，Unity 5秒轮询
-- -------------------------------------------------------------
DROP TABLE IF EXISTS monitor_point_latest;
CREATE TABLE monitor_point_latest (
  point_id    VARCHAR(16)  NOT NULL                COMMENT '点位编号(P01~P04)',
  house_code  VARCHAR(32)  NOT NULL DEFAULT ''     COMMENT '栋舍编号',
  tenant_id   VARCHAR(20)  NOT NULL DEFAULT ''     COMMENT '租户ID',
  temperature DECIMAL(5,2) DEFAULT NULL            COMMENT '温度(℃, 校验-20~60)',
  humidity    DECIMAL(5,2) DEFAULT NULL            COMMENT '湿度(%, 校验0~100)',
  co2         DECIMAL(8,2) DEFAULT NULL            COMMENT 'CO2(ppm, 校验0~10000)',
  ammonia     DECIMAL(6,2) DEFAULT NULL            COMMENT '氨气(ppm, 校验0~200)',
  illuminance DECIMAL(8,2) DEFAULT NULL            COMMENT '扩展指标: 光照 [需与后端实际库核对]',
  sync_time   DATETIME     DEFAULT NULL            COMMENT '采集时间',
  update_time DATETIME     DEFAULT NULL            COMMENT '更新时间(离线判定)',
  PRIMARY KEY (point_id, house_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='监测点最新环境数据';

-- -------------------------------------------------------------
-- 6. monitor_point_history 监测点历史表
-- 用途: 追加保存每次上报，供趋势曲线与报表
-- -------------------------------------------------------------
DROP TABLE IF EXISTS monitor_point_history;
CREATE TABLE monitor_point_history (
  id          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  point_id    VARCHAR(16)  NOT NULL                COMMENT '点位编号',
  house_code  VARCHAR(32)  NOT NULL DEFAULT ''     COMMENT '栋舍编号',
  temperature DECIMAL(5,2) DEFAULT NULL            COMMENT '温度',
  humidity    DECIMAL(5,2) DEFAULT NULL            COMMENT '湿度',
  co2         DECIMAL(8,2) DEFAULT NULL            COMMENT 'CO2',
  ammonia     DECIMAL(6,2) DEFAULT NULL            COMMENT '氨气',
  sync_time   DATETIME     DEFAULT NULL            COMMENT '采集时间',
  PRIMARY KEY (id),
  KEY idx_point_time (point_id, sync_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='监测点环境数据历史';

-- =============================================================
-- 建表完成。初始数据见 data.sql
-- =============================================================
