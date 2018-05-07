'''
简介：
    对数据的分布进行拟合
    即，概率密度函数符合哪种分布！！！ 并不是曲线拟合！！
    通过 *scipy.stats* 和 *powerlaw包* 来实现

方法：
    分布拟合 - FitModel.fit - 基于scipy.stats的连续随机变量的拟合方法
    分布拟合 - FitModel.fit2 - 基于scipy.optimize.curvefit 的 频率密度直方图的数据进行曲线拟合
    绘制拟合 - FitMofel.plot_model

数据：
    data, 为pandas.Series数据类型
    data_pdf，pandas.Series数据类型，index为xdata,values为ydata(概率密度)
        也可以使用data_pdf来做一般性质的曲线拟合


备注：
    使用方法见最下面的example

'''
# test
'''
修改记录：
    * 2017.10.27
        目前只是数据的分布进行拟合，支持的分布需要继续增加

    * 2017.10.30
        拓展曲线拟合，使用Fitmodel(data_pdf)，并不需要曲线拟合功能。

    * 2017.11.1 - 重要
        修复严重错误，关于概率密度的计算，原来采用频率来作为概率
        目前改为np.histgram方法来计算密度
            第二中方法跟R语言中的density方法给出的结果一致，（也是张凌师兄的计算方法）
        第二，采用optimize.curfit方法得到的最优解，跟R语言中的fitdistplus包的结果差别较大
            发现scipy中有更好的曲线拟合方法，scipy.stats中的fit方法,好像数据过大，拟合时间较长
            接下来考虑用scipy.stats中分分布的拟合方法来做
                -存在的一个问题是，stats中所有的分布都采用统一的形式，对于分布的都采用loc,scale来控制，这可能也比较难理解
    * 2017.11.7
        修改了拟合结果,增加了r2的计算
        - - 应该提供保存拟合模型的方法和追加拟合好模型的方法，这样就不用重复拟合工作了
             save_model
             load_model

    * 2017.11.14
        目前基本的分布拟合基本可用,并且支持powerlaw的拟合!

    * 2017.11.20
        目前支持拟合的loglik 和 AIC值的计算了，可以在model中调用!
'''

import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt


class FitModel():
    '''
    model.summary
    （list of dict），summary会用来保存每一次的拟合情况!
    每一次成功的拟合结果包含下面的信息：

    {'method': 拟合的方法（采用fit2还是fit）.
    'dist_name': 拟合分布的名称, str.
    'fit_dist': 拟合用的分布函数.obj
    'para': 拟合的参数结果, list.
    'x_min': 拟合用的x_min.
    'x_max': 拟合用的x_max.
    'pcov': 拟合的方差.
    'r2': R方.
    'LogLik': 似然函数值的对数
    ‘AIC’:拟合模型的AIC值
    'data_pdf': 数据的pdf.
    'xdata': 真实数据，拟合时数据PDF数据的x.
    'ydata': 真实数据，拟合时数据PDF数据的y，,即概率（密度...）.
    'ydata_fit': 拟合数据，拟合出来的ydata, 基于真实数据xdata估计的ydata
    'xdata_plot': 拟合数据，用于绘制模型拟合结果
    'ydata_plot': 拟合数据，用于绘制模型拟合结果}

    '''

    DEFINED_DIST = ['powerlaw',
                    'exponential',
                    'gamma',
                    'lognormal',
                    'weibull',
                    'exponential_powerlaw']

    STATS_DIST = ['lognorm',
                  'expon',
                  'powerlaw',
                  'pareto',
                  'gamma',
                  'exponpow',
                  'norm',
                  'truncexpon',
                  'weibull_min',
                  'weibull_max'
                  ]

    INIT_PARA = {'powerlaw': [1, 1.5],
                 'expon': [2],
                 'gamma': [1, 2],
                 'lognorm': [3, 2],
                 'weibull': [1, 2],
                 'exponpow': [5.3, 1.5, 0.9]}

    # ---------------------------------------------------------
    # 自定义的分布函数
    #
    @staticmethod
    def powerlaw(x, a, beta):
        return a * (x ** (-beta))

    @staticmethod
    def powerlaw_normlized(x, xmin, beta):
        '''
        根据文献定义的标准化的powerlaw函数
        poweRlaw包(R)，powerlaw（python），以及师兄，以及文献：
        - Power laws, Pareto distributions and Zipf's law
        都采用的是这种形式
        '''
        return (beta - 1) / xmin * ((x / xmin) ** (-beta))

    @staticmethod
    def lognorm(x, mu, sigmma):
        return 1 / (x * sigmma * np.sqrt(2 * np.pi)) * np.exp(((np.log(x) - mu) ** 2) / (-2 * sigmma * sigmma))

    @staticmethod
    def expon(x, lam):
        return lam * np.exp(-lam * x)

    @staticmethod
    def weibull(x, alpha, beta):
        '''对比一致
        alpha -- shape para
        beta  -- scale para'''
        return (alpha / beta) * ((x / beta) ** (alpha - 1)) * np.exp(-((x / beta) ** alpha))

    @staticmethod
    def gamma(x, alpha, beta):
        '''检测一致
        alpha 是 shape para
        beta 是 rate para'''
        return ((beta ** alpha) / _gamma(alpha)) * (x ** (alpha - 1)) * np.exp(-beta * x)

    @staticmethod
    def exponpow(x, x0, beta, alpha):
        return (x + x0) ** (-beta) * np.exp(-alpha * x)

    # ---------------------------------------------------------

    def __init__(self, data=None, data_pdf=None, bins=None):
        '''
        :param data: 数据为pd.Series数据
        :param data_pdf: pd.Series, index为xdata，values为probability
        :param bins: 计算pdf时使用的bins数量

        模型的summary会用来保存每一次的拟合情况，一次成功的拟合会被保存成一个字典，包括的信息有：
        res = {'method': 拟合的方法（采用fit2还是fit）.
                'dist_name': 拟合分布的名称,str.
                'fit_dist': 拟合用的分布函数.
                'para': 拟合的参数结果,list.
               'x_min': 拟合用的x_min.
               'x_max':拟合用的x_max.
                'pcov': 拟合的方差.
                'r2': R方.
                'LogLik':Log(似然函数值)
                ‘AIC’:拟合模型的AIC值
                'data_pdf': 数据的pdf.
                'xdata': PDF数据的x.
                'ydata': PDF数据的y，即density.
                'ydata_fit': 拟合出来的ydata.
                'xdata_plot':用于绘制拟合模型的xdata
                'ydata_plot':用于绘制模型拟合的ydata}
          '''

        self.origin_data = data
        if data_pdf is None and data is not None:
            self.bins = bins
            self.data_pdf = FitModel.distribution_pdf(data, bins=bins)
        else:
            self.data_pdf = data_pdf
        self.summary = []

    @staticmethod
    def distribution_fre(data):
        '''
            计算数据的频率密度分布,最后的概率值加起来都等于1
            :param data: list 或者 pandas.Series.
            :return: pandas.Series
            '''
        if data is None:
            return None
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        data_count = data.value_counts().sort_index()
        data_p = data_count / data_count.sum()
        return data_p

    @staticmethod
    def distribution_pdf(data, bins=None):
        '''
        用频率密度直方图来估计概率密度分布
        :param data: 数据
        :return: data_pdf，pandas.Series
        '''
        if data is None:
            return None
        if bins is None:
            bins = 100
        density, xdata = np.histogram(data, bins=bins, density=True)
        xdata = (xdata + np.roll(xdata, -1))[:-1] / 2.0
        data_pdf = pd.Series(density, index=xdata)
        return data_pdf

    @staticmethod
    def distribution_cdf(data, bins=None):
        pdf = FitModel.distribution_pdf(data, bins)
        cdf = []
        for ind in pdf.index:
            cdf.append(pdf[ind:].sum())

        cdf = pd.Series(cdf, index=pdf.index)

        return cdf

    @staticmethod
    def save_model(model, save_path):
        '''保存模型的全部信息，json的形式'''
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(save_path):
        import pickle
        with open(save_path, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def save_result(model, save_path=None):
        '''把模型的拟合参数结果保存成csv文件,方便查看'''

        save_info = []
        for i, each in enumerate(model.summary):
            each_info = []

            each_info.append(('dist_name', each.get('dist_name')))
            for j, para_j in enumerate(each.get('para')):
                each_info.append(('para_' + str(j + 1), para_j))

            others = ['x_min', 'x_max', 'r2', 'LogLik', 'AIC']
            for other_info in others:
                each_info.append((other_info, each.get(other_info)))

            each_info = list(zip(*each_info))
            each_info = pd.Series(each_info[1], index=each_info[0], name=each.get('dist_name'))
            save_info.append(each_info)
        save_info = pd.DataFrame(save_info)
        if save_path:
            save_info.to_csv(save_path, index=True)
            print('--模型已经保存-- %s' % save_path)

    def remove_fitting(self, dist_name):
        '''
        去掉拟合结果中某个fitting结果
        :param dist_name:分布的名称
        :return:去掉的分布的拟合信息
        '''
        summary2 = self.summary[:]
        model_i = None
        for i, each in enumerate(self.summary):
            fitting_dist = each.get('dist_name')
            if fitting_dist == dist_name:
                model_i = each
                summary2.pop(i)
                break
        if len(summary2) < len(self.summary):
            print('- - 删除拟合记录 %s - - ' % dist_name)
        else:
            print('- - 没有拟合记录 %s - -' % dist_name)
        self.summary = summary2

        return model_i

    def r2(self, y, y_fit):
        from sklearn.metrics import r2_score
        '''
        检验过了，符合r2的计算公式
        r2 = 1 - sse/sst
        '''
        r2 = r2_score(y, y_fit)
        return round(r2, 3)

    def calculate_r2(self, y, y_fit):
        y = np.asarray(y)
        y_fit = np.asarray(y_fit)
        mean_y = np.mean(y)
        sse = np.sum(np.power(y - y_fit, 2.0))
        sst = np.sum(np.power(y - mean_y, 2.0))
        r2 = 1 - sse / sst
        return round(r2, 4)

    def fit2(self, distribution, data=None, data_pdf=None,
             x_max=None, x_min=None, initial_para=None, **kwargs):
        '''
        对数据的概率密度分布进行曲线拟合。
        拟合的信息会保存成Dict
        保存到模型的summary 中，以便查询和绘制结果

        :param distribution: 分布名称
        :param data: 需要分布拟合的数据
        :param data_pdf: 概率密度分布数据,Pandas.Series, index为x,values为y
        :param x_max: 拟合分布图像的上限
        :param x_min: 拟合分布图像的下限
        :param initial_para: 拟合分布初始参数
        :return: 拟合的结果，dict
        '''
        if initial_para is None:
            if distribution in FitModel.INIT_PARA.keys():
                initial_para = FitModel.INIT_PARA.get(distribution)
            else:
                print('- - 拟合的分布未定义 - - ')
                return None

        if data_pdf is None:
            if self.data_pdf is not None:
                # self.origin_data不是空的，那么self.data_pdf一定不是空
                data_pdf = self.data_pdf
            else:
                bins = kwargs.get('bins', None)
                if bins is None:
                    bins = self.bins
                data_pdf = FitModel.distribution_pdf(data, bins)
                if data_pdf is None:
                    print('Error: Data is None')
                    return None
                else:
                    self.data_pdf = data_pdf
        else:
            self.data_pdf = data_pdf

        if data is None and self.origin_data is not None:
            data = self.origin_data

        if x_max is not None:
            data_pdf = data_pdf[data_pdf.index < x_max]
            data = data[data < x_max]

        if x_min is not None:
            data_pdf = data_pdf[data_pdf.index > x_min]
            data = data[data > x_min]

        xdata = np.asarray(data_pdf.index.values)
        ydata = np.asarray(data_pdf.values)

        try:
            fit_dist = getattr(FitModel, distribution)
        except AttributeError as e:
            print(' - - FitModel 还不支持分布 ', distribution)
            return None

        print('------------ 拟合分布 %s -------------' % fit_dist)
        para, pcov = optimize.curve_fit(fit_dist, xdata, ydata, p0=initial_para)

        LogLik = np.sum(np.log(fit_dist(data.values, *para)))
        AIC = 2 * len(para) - 2 * LogLik

        ydata_fit = fit_dist(xdata, *para)
        r2 = self.calculate_r2(ydata, ydata_fit)

        plot_xmin = x_min if x_min else data.min()
        plot_xmax = x_max if x_max else data.max()
        xdata_plot = np.linspace(plot_xmin, plot_xmax, 1000)
        ydata_plot = fit_dist(xdata_plot, *para)

        res = {'method': 'FitModel',
               'dist_name': distribution,
               'fit_dist': fit_dist,
               'para': para,
               'x_min': x_min,
               'x_max': x_max,
               'pcov': pcov,
               'r2': r2,
               'LogLik': LogLik,
               'AIC': AIC,
               'data_pdf': data_pdf,
               'xdata': xdata,
               'ydata': ydata,
               'ydata_fit': ydata_fit,
               'xdata_plot': xdata_plot,
               'ydata_plot': ydata_plot
               }
        self.summary.append(res)
        print('- - para - - ', para)
        print('- - r2 - - ', r2)
        return res

    def fit(self, distribution, data=None, x_max=None, x_min=None, **kwargs):
        '''
        :param distribution: 分布的名称，根据scipy提供的连续随机变量确定:
            见https://docs.scipy.org/doc/scipy/reference/stats.html#univariate-and-multivariate-kernel-density-estimation-scipy-stats-kde
        :param data: 拟合使用的数据
        :param x_max: 拟合部分的上界
        :param x_min:拟合部分的下界
        :return: 拟合结果字典
        '''
        try:
            fit_dist = getattr(stats, distribution)
        except AttributeError as e:
            print('- - scipy.satas 不存在分布 - - ', distribution)
            return None

        if data is None and self.origin_data is not None:
            data = self.origin_data

        if x_max is not None:
            data = data[data < x_max]

        if x_min is not None:
            data = data[data > x_min]

        print('------------ 拟合分布 %s -------------' % fit_dist)
        para = fit_dist.fit(data, floc=0)
        arg = para[:-2]
        loc = para[-2]
        scale = para[-1]

        LogLik = np.sum(fit_dist.logpdf(data, *para[:-2], para[-2], para[-1]))
        AIC = 2 * len(para) - 2 * LogLik

        bins = kwargs.get('bins', None)
        if bins is None:
            bins = self.bins
        data_pdf = FitModel.distribution_pdf(data, bins)
        xdata = data_pdf.index.values
        ydata = data_pdf.values

        y_fit = fit_dist.pdf(xdata, *arg, loc=loc, scale=scale)
        r2 = self.calculate_r2(ydata, y_fit)

        plot_xmin = x_min if x_min else data.min()
        plot_xmax = x_max if x_max else data.max()
        xdata_plot = np.linspace(plot_xmin, plot_xmax, 1000)
        ydata_plot = fit_dist.pdf(xdata_plot, *arg, loc=loc, scale=scale)

        res = {'method': 'stats',
               'dist_name': distribution,
               'fit_dist': fit_dist,
               'x_min': x_min,
               'x_max': x_max,
               'para': para,
               'pcov': [],
               'r2': r2,
               'LogLik': LogLik,
               'AIC': AIC,
               'data_pdf': data_pdf,
               'xdata': xdata,
               'ydata': ydata,
               'ydata_fit': y_fit,
               'xdata_plot': xdata_plot,
               'ydata_plot': ydata_plot}

        self.summary.append(res)
        print('- - para - - ', para)
        print('- -  r2  - - ', r2)
        return res

    def fit_powerlaw(self, data=None, x_min=None, x_max=None, use_powerlaw=True,
                     dist_name='powerlaw_nm', **kwargs):
        '''
        主要用powerlaw包来完成对powerlaw的拟合，不指定x_min的话，powerlaw包会找到最佳的x_min
        :param data: pandas.Series数据
        :param x_min: 数据的下界
        :param x_max: 数据的上界
        :param use_powerlaw: 是否使用powerlaw包，否则使用fit2的方法
        :param dist_name: 用来指定此次的拟合名称，方便之后调用，默认为powerlaw_nm（以和powerlaw区分）
        :param kwargs: 其他参数
        :return: 拟合结果的字典
        '''
        if data is None and self.origin_data is not None:
            data = self.origin_data

        if use_powerlaw:
            import powerlaw
            print('- - 使用powerlaw包拟合 - - ')
            model = powerlaw.Fit(data=data.values, xmin=x_min, xmax=x_max)
            XMIN = model.power_law.xmin
            alpha = model.power_law.alpha
            x_min = XMIN
            para = (XMIN, alpha)

            if x_max is not None:
                data = data[data < x_max]

            data = data[data > XMIN]

            LogLik = np.sum(np.log(FitModel.powerlaw_normlized(data.values, *para)))
            AIC = 2 * len(para) - 2 * LogLik

            bins = kwargs.get('bins')
            data_pdf = FitModel.distribution_pdf(data, bins=bins)
            xdata = data_pdf.index.values
            ydata = data_pdf.values

            ydata_fit = FitModel.powerlaw_normlized(xdata, *para)
            r2 = self.calculate_r2(ydata, ydata_fit)

            plot_xmin = x_min if x_min else data.min()
            plot_xmax = x_max if x_max else data.max()
            xdata_plot = np.linspace(plot_xmin, plot_xmax, 1000)
            ydata_plot = FitModel.powerlaw_normlized(xdata_plot, *para)

            res = {'method': 'FitModel',
                   'dist_name': dist_name,
                   'fit_dist': FitModel.powerlaw_normlized,
                   'para': para,
                   'x_min': x_min,
                   'x_max': x_max,
                   'pcov': [],
                   'r2': r2,
                   'LogLik': LogLik,
                   'AIC': AIC,
                   'data_pdf': data_pdf,
                   'xdata': xdata,
                   'ydata': ydata,
                   'ydata_fit': ydata_fit,
                   'xdata_plot': xdata_plot,
                   'ydata_plot': ydata_plot}

            self.summary.append(res)
            print('- - para - - ', para)
            print('- - r2 - - ', r2)
            return res
        else:
            self.fit2('powerlaw', data, x_max=x_max, x_min=x_min, **kwargs)

    def plot_model(self, style=0, axes=None, log_log=True,
                   mfrow=None, plot_origindata=True, **kwargs):
        '''

        :param style: 绘制风格
        :param axes: ax对象
        :param log_log: 是否为log-log
        :param mfrow: 图像分割,只在style为1时使用
        :param plot_origindata: 是否绘制原数据的数据
        :param kwargs:
        :return: fig,ax 或 ax
        '''
        if len(self.summary) < 1:
            print('- - 模型还没有拟合 - - ')
            return None

        if style == 0:
            if axes is None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
            else:
                ax = axes
            for model in self.summary:
                xdata_plot = model.get('xdata_plot')
                ydata_fit = model.get('ydata_plot')
                distribution = model.get('dist_name')
                r2 = model.get('r2')
                legend_label = distribution + ' (r2: %.3f)' % r2
                ax.plot(xdata_plot, ydata_fit, label=legend_label)
                ax.legend()
            if plot_origindata:
                ax.plot(self.data_pdf.index.values, self.data_pdf.values,
                        linestyle='', marker='o', mfc='#2E68AA', mec='#2E68AA')
            else:
                ax.plot(model.get('xdata'), model.get('ydata'), linestyle='', marker='o',
                        mfc='#2E68AA', mec='#2E68AA')
            ax.set_ylabel('Prob')
            if log_log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            if 'xlim' in kwargs.keys():
                ax.set_xlim(kwargs.get('xlim'))
            if 'ylim' in kwargs.keys():
                ax.set_ylim(kwargs.get('ylim'))

        if style == 1:
            model_num = len(self.summary)
            if mfrow is None:
                # 设置图片分割方式，如果没有的话
                if model_num < 3:
                    mfrow = (1, model_num)
                elif model_num < 7:
                    mfrow = (2, 3)
                else:
                    mfrow = (3, 3)
            # 开始绘制每一个ax
            if axes is None:
                fig = plt.figure()
                ax = [fig.add_subplot(mfrow[0], mfrow[1], i + 1) for i in len(self.summary)]
            else:
                ax = axes
            for i, model in enumerate(self.summary):
                xdata_plot = model.get('xdata_plot')
                ydata_fit = model.get('ydata_plot')

                distribution = model.get('dist_name')
                r2 = model.get('r2')
                legend_label = distribution + ' (r2: %.3f)' % r2
                ax[i].plot(xdata_plot, ydata_fit, label=legend_label)
                if plot_origindata:
                    ax[i].plot(self.data_pdf.index.values, self.data_pdf.values,
                               linestyle='', marker='o', mfc='', mec='#2E68AA')
                else:
                    ax[i].plot(model.get('xdata'), model.get('ydata'), linestyle='', marker='o',
                               mfc='#2E68AA', mec='#2E68AA')
                ax[i].legend()
                ax[i].set_ylabel('Prob')
                if log_log:
                    ax[i].set_yscale('log')
                    ax[i].set_xscale('log')
                if 'xlim' in kwargs.keys():
                    ax[i].set_xlim(kwargs.get('xlim'))
                if 'ylim' in kwargs.keys():
                    ax[i].set_ylim(kwargs.get('ylim'))

        if not axes:
            return fig, ax
        else:
            return ax


def example_fitting():
    import os.path

    # 读取数据
    DataDir = r'G:\data'
    path_dis = os.path.join(DataDir, 'DistanceAC.csv')
    data = pd.read_csv(path_dis, header=0)
    data = data['DistanceAC']
    data = data[data > 0]

    # 建立模型进行数据分布的拟合
    # fit方法会返回本次拟合的结果的dict
    model = FitModel(data=data)
    model_expon = model.fit(distribution='expon', x_max=8)
    # model_expon就是拟合expon的结果，包含的信息见model.summary介绍

    model.fit(distribution='lognorm')
    model.fit(distribution='gamma')
    model.fit_powerlaw(x_min=4, dist_name='Powerlaw_tail_4')
    model.plot_model(style=1)

    # 保存模型
    save_path = os.path.join(DataDir, 'model.json')
    FitModel.save_model(model, save_path)

    # 读取已经保存的模型
    model_2 = FitModel.load_model(save_path)

    # 查看各种拟合的结果
    for each in model.summary:
        print('-------------------------------------------')
        print(' -dist- ', each['dist_name'], ' -x_min- ', each['x_min'])
        print(' -para- ', each['para'], ' -r2- ', each['r2'])
        print(' -LogLik- ', each['LogLik'], ' -AIC- ', each['AIC'])

    # 去掉powerlaw这个结果，原因：例如想单独绘制这个拟合
    model_power = model.remove_fitting('Powerlaw_tail_4')

    # 单独绘制powerlaw
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(model_power['xdata_plot'], model_power['ydata_plot'], label='Powerlaw_tail_4')
    plt.show()


def compare_fitting_powerlaw():
    '''
    对比powerlaw包的拟合结果

    ** 对于powerlaw的拟合，最主要的部分在于确定尾部的开始，即xmin
        方法1时，肉眼观测
        方法2是使用powerlaw包，可以确定最佳的xmin，但是拟合的部分会在原数据的pdf上方（density高一点）
        方法3是使用powerlaw确定的xmin，用FitModel.fit2('powerlaw')，来拟合aplha，
            这样可以时拟合结果绘制时跟原数据一致
            差异见example_powerlaw.png

        * 所以结论是，但你只需要绘制尾部的时候，直接采用powerlaw拟合
        * 但需要观测全部的数据分布，然后powerlaw在尾部时，使用方法3绘制到原来的数据分布曲线上。
    '''
    import powerlaw
    distance = pd.Series()
    data_pdf = FitModel.distribution_pdf(distance, bins=100)

    # -----------powerlaw包的拟合方法----------------
    result_1 = powerlaw.Fit(distance.values)
    XMIN_FOUND = result_1.power_law.xmin
    print('alpha: ', result_1.power_law.alpha)
    print('xmin:', XMIN_FOUND)

    # ------------根据自己定义的做拟合，对比使用----------
    model_2 = FitModel(data=distance, bins=100)
    result_2 = model_2.fit2('powerlaw', x_min=XMIN_FOUND)

    plt.plot(data_pdf.index.values, data_pdf.values, 'ko', label='all data distribution')
    # 绘制powerlaw包拟合的结果

    # -------------powerlaw的结果
    tail_data_pdf = FitModel.distribution_pdf(distance[distance > XMIN_FOUND], bins=100)
    fit_powerlaw_package = FitModel.powerlaw_normlized(tail_data_pdf.index.values, XMIN_FOUND, result_1.power_law.alpha)
    plt.plot(tail_data_pdf.index.values, fit_powerlaw_package, 'b-', linewidth=3,
             label='powerlaw_package_fit(normalized aplha:%.3f))' % result_1.power_law.alpha)
    plt.plot(tail_data_pdf.index.values, tail_data_pdf, 'go', label='tail data_pdf')

    # -------------fit2方法的结果
    plt.plot(result_2['xdata'], result_2['ydata_fit'], 'r-', linewidth=3,
             label='defined_powerlaw_fit(C:%.3f,alpha:%.3f)' % (result_2['para'][0], result_2['para'][1]))
    plt.plot(result_2['xdata'], result_2['ydata'], 'bo', label='data for fitting')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    example_fitting()
    # compare_fitting_powerlaw()



